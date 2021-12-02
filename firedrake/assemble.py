import abc
from collections import OrderedDict
from enum import IntEnum, auto
import functools
import itertools
import operator

import cachetools
import finat
import firedrake
import numpy
from tsfc import kernel_args
from tsfc.finatinterface import create_element
import ufl
from firedrake import (assemble_expressions, extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.functionspacedata import entity_dofs_key, entity_permutations_key
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.slate.slac.kernel_builder import CellFacetKernelArg, LayerCountKernelArg, LayerKernelArg
from firedrake.utils import ScalarType, tuplify
from pyop2 import op2
import pyop2.wrapper_kernel
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = ("AssemblyType", "assemble")


class AssemblyType(IntEnum):
    """Enum enumerating possible assembly types.

    See ``"assembly_type"`` from :func:`assemble` for more information.
    """
    SOLUTION = auto()
    RESIDUAL = auto()


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
        a :class:`~slate.TensorBase` expression.
    :arg tensor: Existing tensor object to place the result in.
    :arg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
    TODO update here
    :kwarg assembly_type: String indicating how boundary conditions are applied
        (may be ``"solution"`` or ``"residual"``). If ``"solution"`` then the
        boundary conditions are applied as expected whereas ``"residual"`` zeros
        the selected components of the tensor.
    :kwarg form_compiler_parameters: Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a :class:`~ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        :class:`~ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    :kwarg mat_type: String indicating how a 2-form (matrix) should be
        assembled -- either as a monolithic matrix (``"aij"`` or ``"baij"``),
        a block matrix (``"nest"``), or left as a :class:`.ImplicitMatrix` giving
        matrix-free actions (``'matfree'``). If not supplied, the default value in
        ``parameters["default_matrix_type"]`` is used.  BAIJ differs
        from AIJ in that only the block sparsity rather than the dof
        sparsity is constructed.  This can result in some memory
        savings, but does not work with all PETSc preconditioners.
        BAIJ matrices only make sense for non-mixed matrices.
    :kwarg sub_mat_type: String indicating the matrix type to
        use *inside* a nested block matrix.  Only makes sense if
        ``mat_type`` is ``nest``.  May be one of ``"aij"`` or ``"baij"``.  If
        not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :kwarg appctx: Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    :kwarg options_prefix: PETSc options prefix to apply to matrices.

    :returns: See below.

    If expr is a :class:`~ufl.classes.Form` or Slate tensor expression then
    this evaluates the corresponding integral(s) and returns a :class:`float`
    for 0-forms, a :class:`.Function` for 1-forms and a :class:`.Matrix` or
    :class:`.ImplicitMatrix` for 2-forms. In the case of 2-forms the rows
    correspond to the test functions and the columns to the trial functions.

    If expr is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``expr`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``expr`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    .. note::
        For 1-form assembly, the resulting object should in fact be a *cofunction*
        instead of a :class:`.Function`. However, since cofunctions are not
        currently supported in UFL, functions are used instead.
    """
    # TODO It bothers me that we use the same code for two types of expression. Ideally
    # we would define a shared interface for the two to follow. Otherwise I feel like
    # having assemble_form and assemble_slate as separate functions would be desirable.
    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        return _assemble_form(expr, *args, **kwargs)
    elif isinstance(expr, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(expr)
    else:
        raise TypeError(f"Unable to assemble: {expr}")


@PETSc.Log.EventDecorator()
def allocate_matrix(
    expr,
    bcs=None,
    *,
    mat_type=None,
    sub_mat_type=None,
    appctx=None,
    form_compiler_parameters=None,
    options_prefix=None
):
    r"""Allocate a matrix given an expression.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    bcs = bcs or ()
    appctx = appctx or {}

    matfree = mat_type == "matfree"
    arguments = expr.arguments()
    if bcs is None:
        bcs = ()
    else:
        if any(isinstance(bc, EquationBC) for bc in bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
    if matfree:
        return matrix.ImplicitMatrix(expr, bcs,
                                     appctx=appctx,
                                     fc_params=form_compiler_parameters,
                                     options_prefix=options_prefix)

    integral_types = set(i.integral_type() for i in expr.integrals())
    for bc in bcs:
        integral_types.update(integral.integral_type()
                              for integral in bc.integrals())
    nest = mat_type == "nest"
    if nest:
        baij = sub_mat_type == "baij"
    else:
        baij = mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and mat_type == "baij":
        raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

    get_cell_map = operator.methodcaller("cell_node_map")
    get_extf_map = operator.methodcaller("exterior_facet_node_map")
    get_intf_map = operator.methodcaller("interior_facet_node_map")
    domains = OrderedDict((k, set()) for k in (get_cell_map,
                                               get_extf_map,
                                               get_intf_map))
    mapping = {"cell": (get_cell_map, op2.ALL),
               "exterior_facet_bottom": (get_cell_map, op2.ON_BOTTOM),
               "exterior_facet_top": (get_cell_map, op2.ON_TOP),
               "interior_facet_horiz": (get_cell_map, op2.ON_INTERIOR_FACETS),
               "exterior_facet": (get_extf_map, op2.ALL),
               "exterior_facet_vert": (get_extf_map, op2.ALL),
               "interior_facet": (get_intf_map, op2.ALL),
               "interior_facet_vert": (get_intf_map, op2.ALL)}
    for integral_type in integral_types:
        try:
            get_map, region = mapping[integral_type]
        except KeyError:
            raise ValueError(f"Unknown integral type '{integral_type}'")
        domains[get_map].add(region)

    test, trial = arguments
    map_pairs, iteration_regions = zip(*(((get_map(test), get_map(trial)),
                                          tuple(sorted(regions)))
                                         for get_map, regions in domains.items()
                                         if regions))
    try:
        sparsity = op2.Sparsity((test.function_space().dof_dset,
                                 trial.function_space().dof_dset),
                                tuple(map_pairs),
                                iteration_regions=tuple(iteration_regions),
                                nest=nest,
                                block_sparse=baij)
    except SparsityFormatError:
        raise ValueError("Monolithic matrix assembly not supported for systems "
                         "with R-space blocks")

    return matrix.Matrix(expr, bcs, mat_type, sparsity, ScalarType,
                         options_prefix=options_prefix)


@PETSc.Log.EventDecorator()
def create_assembly_callable(expr, tensor=None, bcs=None, form_compiler_parameters=None,
                             mat_type=None, sub_mat_type=None, diagonal=False):
    r"""Create a callable object than be used to assemble expr into a tensor.

    This is really only designed to be used inside residual and
    jacobian callbacks, since it always assembles back into the
    initially provided tensor.  See also :func:`allocate_matrix`.

    .. warning::

        This function is now deprecated.

    .. warning::

       Really do not use this function unless you know what you're doing.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("once", DeprecationWarning)
        warnings.warn("create_assembly_callable is now deprecated. Please use assemble instead.",
                      DeprecationWarning)

    if tensor is None:
        raise ValueError("Have to provide tensor to write to")
    return functools.partial(assemble, expr,
                             tensor=tensor,
                             bcs=bcs,
                             form_compiler_parameters=form_compiler_parameters,
                             mat_type=mat_type,
                             sub_mat_type=sub_mat_type,
                             diagonal=diagonal,
                             assembly_type=AssemblyType.SOLUTION)


class _FormAssembler(abc.ABC):

    def __init__(self, form, form_compiler_parameters=None):
        self._form = form
        self._form_compiler_params = form_compiler_parameters

    @abc.abstractproperty
    def result(self):
        ...

    @abc.abstractproperty
    def bcs(self):
        ...

    def assemble(self):
        self.assemble_inner(self._form, self.bcs)
        return self.result

    @abc.abstractproperty
    def diagonal(self):
        ...

    def compile_form(self, form=None):
        form = form or self._form

        try:
            topology, = set(d.topology for d in form.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        for o in itertools.chain(form.arguments(), form.coefficients()):
            domain = o.ufl_domain()
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(form, ufl.Form):
            return tsfc_interface.compile_form(form, "form", diagonal=self.diagonal,
                                               parameters=self._form_compiler_params)
        elif isinstance(form, slate.TensorBase):
            return slac.compile_expression(form, compiler_parameters=self._form_compiler_params)
        else:
            raise AssertionError

    def assemble_inner(self, form, bcs):
        tsfc_knls = self.compile_form(form)
        all_integer_subdomain_ids = tsfc_interface.gather_integer_subdomain_ids(tsfc_knls)

        knls = [_make_wrapper_kernel(form, tsfc_knl, all_integer_subdomain_ids,
                                     diagonal=self.diagonal,
                                     unroll=self.needs_unrolling(tsfc_knl, bcs))
                for tsfc_knl in tsfc_knls]

        for tknl, knl in zip(tsfc_knls, knls):
            _execute_parloop(form, tknl, knl, all_integer_subdomain_ids, tensor=self._tensor,
                             diagonal=self.diagonal,
                             lgmaps=self.collect_lgmaps(tknl, bcs))

        for bc in bcs:
            if isinstance(bc, EquationBC):
                bc = bc.extract_form("F")
            self._apply_bc(bc)

    def needs_unrolling(self, knl, bcs):
        return False

    def collect_lgmaps(self, knl, bcs):
        return None


class _ZeroFormAssembler(_FormAssembler):

    bcs = ()
    """Boundary conditions are not compatible with zero forms."""

    diagonal = False
    """Diagonal assembly not possible for zero forms."""

    def __init__(self, form, **kwargs):
        super().__init__(form, **kwargs)

        if len(form.arguments()) != 0:
            raise ValueError("Cannot assemble a 0-form with arguments")

        self._tensor = op2.Global(1, [0.0], dtype=utils.ScalarType)

    @property
    def result(self):
        return self._tensor.data[0]


class _OneFormAssembler(_FormAssembler):

    def __init__(self, form, tensor=None, bcs=None, *,
                 assembly_type=AssemblyType.SOLUTION,
                 diagonal=False, **kwargs):
        super().__init__(form, **kwargs)

        if diagonal:
            test, trial = form.arguments()
            if test.function_space() != trial.function_space():
                raise ValueError("Can only assemble the diagonal of 2-form if the "
                                 "function spaces match")
        else:
            test, = form.arguments()

        if tensor:
            if test.function_space() != tensor.function_space():
                raise ValueError("Form's argument does not match provided result tensor")
            tensor.dat.zero()
        else:
            tensor = firedrake.Function(test.function_space())

        self._bcs = bcs
        self._tensor = tensor
        self._assembly_type = assembly_type
        self._diagonal = diagonal

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def bcs(self):
        return self._bcs

    @property
    def result(self):
        return self._tensor

    def _apply_bc(self, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            if self._diagonal:
                raise NotImplementedError("Diagonal assembly and EquationBC not supported")
            bc.zero(self._tensor)
            self.assemble_inner(bc.f, bc.bcs)
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        if self._assembly_type == AssemblyType.SOLUTION:
            if self._diagonal:
                bc.set(self._tensor, 1)
            else:
                bc.apply(self._tensor)
        elif self._assembly_type == AssemblyType.RESIDUAL:
            bc.zero(self._tensor)
        else:
            raise AssertionError


class _TwoFormAssembler(_FormAssembler):

    diagonal = False
    """Diagonal assembly not possible for two forms."""

    def __init__(self, form, tensor=None, bcs=None, *,
                 mat_type, sub_mat_type, form_compiler_parameters,
                 options_prefix):
        super().__init__(form, form_compiler_parameters=form_compiler_parameters)

        mat_type, sub_mat_type = self._get_mat_type(mat_type, sub_mat_type, form.arguments())

        assert mat_type != "matfree"

        if tensor:
            tensor.M.zero()
        else:
            tensor = allocate_matrix(
                form, bcs, mat_type=mat_type, sub_mat_type=sub_mat_type,
                form_compiler_parameters=form_compiler_parameters,
                options_prefix=options_prefix
            )

        self._tensor = tensor
        self._bcs = bcs

    @property
    def bcs(self):
        return self._bcs

    @property
    def result(self):
        self._tensor.M.assemble()
        return self._tensor

    def _needs_unrolling(self, all_bcs, Vrow, Vcol, i, j):
        if len(Vrow) > 1:
            bcrow = tuple(bc for bc in all_bcs
                          if bc.function_space_index() == i)
        else:
            bcrow = all_bcs
        if len(Vcol) > 1:
            bccol = tuple(bc for bc in all_bcs
                          if bc.function_space_index() == j
                          and isinstance(bc, DirichletBC))
        else:
            bccol = tuple(bc for bc in all_bcs
                          if isinstance(bc, DirichletBC))
        return any(bc.function_space().component is not None
                   for bc in itertools.chain(bcrow, bccol))

    def needs_unrolling(self, knl, bcs):
        test, trial = self._form.arguments()
        Vrow = test.function_space()
        Vcol = trial.function_space()
        row, col = knl.indices
        if row is None and col is None:
            return any(self._needs_unrolling(bcs, Vrow, Vcol, i, j)
                                   for i, j in numpy.ndindex(self._tensor.block_shape))
        else:
            assert row is not None and col is not None
            return self._needs_unrolling(bcs, Vrow, Vcol, row, col)

    def collect_lgmaps(self, knl, bcs):
        test, trial = self._form.arguments()
        Vrow = test.function_space()
        Vcol = trial.function_space()
        row, col = knl.indices
        if row is None and col is None:
            return tuple(self._collect_lgmaps(bcs, Vrow, Vcol, i, j)
                                   for i, j in numpy.ndindex(self._tensor.block_shape))
        else:
            assert row is not None and col is not None
            return self._collect_lgmaps(bcs, Vrow, Vcol, row, col)

    def _collect_lgmaps(self, all_bcs, Vrow, Vcol, i, j):
        if len(Vrow) > 1:
            bcrow = tuple(bc for bc in all_bcs
                          if bc.function_space_index() == i)
        else:
            bcrow = all_bcs
        if len(Vcol) > 1:
            bccol = tuple(bc for bc in all_bcs
                          if bc.function_space_index() == j
                          and isinstance(bc, DirichletBC))
        else:
            bccol = tuple(bc for bc in all_bcs
                          if isinstance(bc, DirichletBC))
        rlgmap, clgmap = self._tensor.M[i, j].local_to_global_maps
        rlgmap = Vrow[i].local_to_global_map(bcrow, lgmap=rlgmap)
        clgmap = Vcol[j].local_to_global_map(bccol, lgmap=clgmap)
        return rlgmap, clgmap

    def _apply_bc(self, bc):
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            self.assemble_inner(bc.f, bc.bcs)
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        op2tensor = self._tensor.M
        shape = tuple(len(a.function_space()) for a in self._tensor.a.arguments())

        V = bc.function_space()
        nodes = bc.nodes
        for i, j in numpy.ndindex(shape):
            # Set diagonal entries on bc nodes to 1 if the current
            # block is on the matrix diagonal and its index matches the
            # index of the function space the bc is defined on.
            if i != j:
                continue
            if V.component is None and V.index is not None:
                # Mixed, index (no ComponentFunctionSpace)
                if V.index == i:
                    op2tensor[i, j].set_local_diagonal_entries(nodes)
            elif V.component is not None:
                # ComponentFunctionSpace, check parent index
                if V.parent.index is not None:
                    # Mixed, index doesn't match
                    if V.parent.index != i:
                        continue
                    # Index matches
                op2tensor[i, j].set_local_diagonal_entries(nodes, idx=V.component)
            elif V.index is None:
                op2tensor[i, j].set_local_diagonal_entries(nodes)
            else:
                raise RuntimeError("Unhandled BC case")

    @staticmethod
    def _get_mat_type(mat_type, sub_mat_type, arguments):
        """Validate the matrix types provided by the user and set any that are
        undefined to default values.

        :arg mat_type: (:class:`str`) PETSc matrix type for the assembled matrix.
        :arg sub_mat_type: (:class:`str`) PETSc matrix type for blocks if
            ``mat_type`` is ``"nest"``.
        :arg arguments: The test and trial functions of the expression being assembled.
        :raises ValueError: On bad arguments.
        :returns: 2-:class:`tuple` of validated/default ``mat_type`` and ``sub_mat_type``.
        """
        if mat_type is None:
            mat_type = parameters.parameters["default_matrix_type"]
            if any(V.ufl_element().family() == "Real"
                   for arg in arguments
                   for V in arg.function_space()):
                mat_type = "nest"
        if mat_type not in {"matfree", "aij", "baij", "nest", "dense"}:
            raise ValueError(f"Unrecognised matrix type, '{mat_type}'")
        if sub_mat_type is None:
            sub_mat_type = parameters.parameters["default_sub_matrix_type"]
        if sub_mat_type not in {"aij", "baij"}:
            raise ValueError(f"Invalid submatrix type, '{sub_mat_type}' (not 'aij' or 'baij')")
        return mat_type, sub_mat_type


def _assemble_form(form, tensor=None, bcs=None, *,
                   assembly_type=AssemblyType.SOLUTION,
                   diagonal=False,
                   mat_type=None,
                   sub_mat_type=None,
                   appctx=None,
                   options_prefix=None,
                   form_compiler_parameters=None):
    """Assemble a form.

    :arg form:
        The :class:`~ufl.classes.Form` or :class:`~slate.TensorBase` to be assembled.
    :args args:
        Extra positional arguments to pass to the underlying :class:`_Assembler` instance.
        See :func:`assemble` for more information.
    :kwarg diagonal:
        Flag indicating whether or not we are assembling the diagonal of a matrix.
    :kwargs kwargs:
        Extra keyword arguments to pass to the underlying :class:`_Assembler` instance.
        See :func:`assemble` for more information.
    """
    # Ensure mesh is 'initialised' as we could have got here without building a
    # function space (e.g. if integrating a constant).
    for mesh in form.ufl_domains():
        mesh.init()

    bcs = solving._extract_bcs(bcs)

    rank = len(form.arguments())
    if rank == 0:
        assert tensor is None and bcs == ()
        return _ZeroFormAssembler(form, form_compiler_parameters=form_compiler_parameters).assemble()
    elif rank == 1 or (rank == 2 and diagonal):
        return _OneFormAssembler(form, tensor, bcs, assembly_type=assembly_type,
                                 diagonal=diagonal, form_compiler_parameters=form_compiler_parameters).assemble()
    elif rank == 2:
        if tensor is not None:
            if tensor.a.arguments() != form.arguments():
                raise ValueError("Form's arguments do not match provided result tensor")
        # We do something radically different form matrix free - intercept here
        if mat_type == "matfree":
            if tensor is None:
                tensor = allocate_matrix(
                    form, bcs, mat_type=mat_type, sub_mat_type=sub_mat_type, appctx=appctx,
                    form_compiler_parameters=form_compiler_parameters,
                    options_prefix=options_prefix
                )
            tensor.assemble()
            return tensor
        else:
            assembler = _TwoFormAssembler(form, tensor, bcs,
                                          mat_type=mat_type,
                                          sub_mat_type=sub_mat_type,
                                          form_compiler_parameters=form_compiler_parameters,
                                          options_prefix=options_prefix)
            return assembler.assemble()
    else:
        raise AssertionError


class _AssembleWrapperKernelBuilder:

    def __init__(self, expr, split_knl, all_integer_subdomain_ids, *, diagonal=False, unroll=False):
        """TODO

        .. note::

            expr should work even if it is 'pure UFL'.
        """
        self._expr = expr
        self._indices, self._kinfo = split_knl
        self._diagonal = diagonal
        self._unroll = unroll
        self.all_integer_subdomain_ids = all_integer_subdomain_ids.get(self._kinfo.integral_type, None)

    def build(self):
        domain = self.mesh
        extruded = domain.extruded
        constant_layers = extruded and not domain.variable_layers

        subset = self.subset()

        self.extruded = extruded

        # Icky generator so we can access the correct coefficients in order
        def coeffs():
            for n, split_map in self._kinfo.coefficient_map:
                c = self._expr.coefficients()[n]
                split_c = c.split()
                for c_ in (split_c[i] for i in split_map):
                    yield c_
        self.coeffs_iterator = iter(coeffs())

        self.integral_type = self._kinfo.integral_type

        wrapper_kernel_args = [
            self._as_wrapper_kernel_arg(arg)
            for arg in self._kinfo.tsfc_kernel_args
            if arg.intent is not None
        ]

        iteration_region = {
            "exterior_facet_top": op2.ON_TOP,
            "exterior_facet_bottom": op2.ON_BOTTOM,
            "interior_facet_horiz": op2.ON_INTERIOR_FACETS
        }.get(self._kinfo.integral_type, None)

        return op2.WrapperKernel(
            self._kinfo.kernel,
            wrapper_kernel_args,
            iteration_region=iteration_region,
            pass_layer_arg=self._kinfo.pass_layer_arg,
            extruded=extruded,
            constant_layers=constant_layers,
            subset=subset
        )

    # TODO I copy this for parloops too
    @functools.cached_property
    def mesh(self):
        return self._expr.ufl_domains()[self._kinfo.domain_number]

    def subset(self):
        # Assume that if subdomain_data is not None then we are dealing with
        # a subset. This is potentially a bit dodgy.
        subdomain_data = self._expr.subdomain_data()[self.mesh].get(self._kinfo.integral_type, None)
        if subdomain_data is not None:
            return True

        if self._kinfo.subdomain_id == "everywhere":
            return False
        elif self._kinfo.subdomain_id == "otherwise":
            return self.all_integer_subdomain_ids is not None
        else:
            return True

    def _as_wrapper_kernel_arg(self, tsfc_arg):
        # TODO Make singledispatchmethod with Python 3.8
        return _as_wrapper_kernel_arg(tsfc_arg, self)

    def get_dim_and_map(self, finat_element):
        map_id = self._get_map_id(finat_element)

        # offset only valid for extruded
        if self.extruded:
            offset = tuple(eutils.calculate_dof_offset(finat_element))
            # For interior facet integrals we double the size of the offset array
            if self.integral_type in {"interior_facet", "interior_facet_vert"}:
                offset += offset
        else:
            offset = None

        handler = _ElementHandler(finat_element)
        dim = handler.tensor_shape
        arity = handler.node_shape
        if self.integral_type in {"interior_facet", "interior_facet_vert"}:
            arity *= 2

        map_arg = op2.MapWrapperKernelArg(map_id, arity, offset)
        return dim, map_arg

    def get_function_spaces(self, arguments):
        indices = self._indices
        if all(i is None for i in indices):
            return tuple(a.ufl_function_space() for a in arguments)
        elif all(i is not None for i in indices):
            return tuple(a.ufl_function_space()[i] for i, a in zip(indices, arguments))
        else:
            raise AssertionError

    def make_arg(self, elements):
        if len(elements) == 1:
            func = self._make_dat_wrapper_kernel_arg
        elif len(elements) == 2:
            func = self._make_mat_wrapper_kernel_arg
        else:
            raise AssertionError
        return func(*[e._elem for e in elements])

    def make_mixed_arg(self, elements):
        subargs = []
        for splitstuff in itertools.product(*[e.split() for e in elements]):
            subargs.append(self.make_arg(splitstuff))

        if len(elements) == 1:
            return op2.MixedDatWrapperKernelArg(tuple(subargs))
        elif len(elements) == 2:
            e1, e2 = elements
            shape = len(e1.split()), len(e2.split())
            return op2.MixedMatWrapperKernelArg(tuple(subargs), shape)
        else:
            raise AssertionError

    def _make_dat_wrapper_kernel_arg(self, finat_element):
        dim, map_arg = self.get_dim_and_map(finat_element)
        return op2.DatWrapperKernelArg(dim, map_arg)

    def _make_mat_wrapper_kernel_arg(self, relem, celem):
        rdim, rmap_arg = self.get_dim_and_map(relem)
        cdim, cmap_arg = self.get_dim_and_map(celem)

        # PyOP2 matrix objects have scalar dims so we cope with that here...
        rdim = (numpy.prod(rdim, dtype=int),)
        cdim = (numpy.prod(cdim, dtype=int),)

        return op2.MatWrapperKernelArg(((rdim+cdim,),), (rmap_arg, cmap_arg), unroll=self._unroll)

    @staticmethod
    def _get_map_id(finat_element):
        """Return a key that is used to check if we reuse maps.

        functionspacedata.py does the same thing.
        """
        # TODO need to look at measure...
        # functionspacedata does some magic and replaces tensorelements with base
        if isinstance(finat_element, finat.TensorFiniteElement):
            finat_element = finat_element.base_element

        real_tensorproduct = eutils.is_real_tensor_product_element(finat_element)

        entity_dofs = finat_element.entity_dofs()
        try:
            eperm_key = entity_permutations_key(finat_element.entity_permutations)
        except NotImplementedError:
            eperm_key = None
        return entity_dofs_key(entity_dofs), real_tensorproduct, eperm_key


@functools.singledispatch
def _as_wrapper_kernel_arg(tsfc_arg, self):
    raise NotImplementedError


@_as_wrapper_kernel_arg.register(kernel_args.OutputKernelArg)
def _as_wrapper_kernel_arg_output(_, self):
    arguments = self._expr.arguments()

    # lower this
    if len(arguments) == 0:
        return op2.GlobalWrapperKernelArg((1,))

    if self._diagonal:
        test, trial = arguments
        arguments = test,

    function_spaces = self.get_function_spaces(arguments)
    # need to drop real elements here since they correspond to global blocks
    # (and the data structure loses a rank)
    elems = [_ElementHandler(create_element(V.ufl_element()))
             for V in function_spaces
             if V.ufl_element().family() != "Real"]

    if len(elems) == 0:
        return op2.GlobalWrapperKernelArg((1,))

    if any(e.is_mixed for e in elems):
        return self.make_mixed_arg(elems)
    else:
        return self.make_arg(elems)


@_as_wrapper_kernel_arg.register(kernel_args.CoordinatesKernelArg)
def _as_wrapper_kernel_arg_coordinates(_, self):
    domain = self.mesh
    finat_element = create_element(domain.ufl_coordinate_element())
    return self._make_dat_wrapper_kernel_arg(finat_element)


@_as_wrapper_kernel_arg.register(kernel_args.ConstantKernelArg)
def _as_wrapper_kernel_arg_constant(_, self):
    coeff = next(self.coeffs_iterator)
    ufl_element = coeff.ufl_function_space().ufl_element()
    assert ufl_element.family() == "Real"
    return op2.GlobalWrapperKernelArg((ufl_element.value_size(),))


@_as_wrapper_kernel_arg.register(kernel_args.CoefficientKernelArg)
def _as_wrapper_kernel_arg_coefficient(_, self):
    coeff = next(self.coeffs_iterator)
    finat_element = create_element(coeff.ufl_function_space().ufl_element())
    return self._make_dat_wrapper_kernel_arg(finat_element)


@_as_wrapper_kernel_arg.register(kernel_args.CellSizesKernelArg)
def _as_wrapper_kernel_arg_cell_sizes(_, self):
    domain = self.mesh
    # See set_cell_sizes from tsfc.kernel_interface.firedrake_loopy
    ufl_element = ufl.FiniteElement("P", domain.ufl_cell(), 1)
    finat_element = create_element(ufl_element)
    return self._make_dat_wrapper_kernel_arg(finat_element)


@_as_wrapper_kernel_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_wrapper_kernel_arg_exterior_facet(_, self):
    return op2.DatWrapperKernelArg((1,))


@_as_wrapper_kernel_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_wrapper_kernel_arg_interior_facet(_, self):
    return op2.DatWrapperKernelArg((2,))


@_as_wrapper_kernel_arg.register(CellFacetKernelArg)
def _as_wrapper_kernel_arg_cell_facet(_, self):
    # TODO Share this functionality with Slate kernel_builder.py
    if self.mesh.extruded:
        # TODO This is not sufficiently stripped
        num_facets = self.mesh._base_mesh.ufl_cell().num_facets()
    else:
        num_facets = self.mesh.ufl_cell().num_facets()
    return op2.DatWrapperKernelArg((num_facets, 2))


@_as_wrapper_kernel_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_wrapper_kernel_arg_cell_orientations(_, self):
    # this is taken largely from mesh.py where we observe that the function space is
    # DG0.
    ufl_element = ufl.FiniteElement("DG", cell=self._expr.ufl_domain().ufl_cell(), degree=0)
    finat_element = create_element(ufl_element)
    return self._make_dat_wrapper_kernel_arg(finat_element)


@_as_wrapper_kernel_arg.register(LayerCountKernelArg)
def _as_wrapper_kernel_arg_layer_count(_, self):
    return op2.GlobalWrapperKernelArg((1,))


class _ElementHandler:

    def __init__(self, elem):
        self._elem = elem

    @property
    def node_shape(self):
        if self._is_tensor_element:
            shape = self._elem.index_shape[:-len(self.tensor_shape)]
        else:
            shape = self._elem.index_shape

        shape = numpy.prod(shape, dtype=int)
        return shape

    @property
    def tensor_shape(self):
        return self._elem._shape if self._is_tensor_element else (1,)

    @property
    def is_mixed(self):
        return isinstance(self._elem, finat.EnrichedElement) and self._elem.is_mixed

    def split(self):
        if not self.is_mixed:
            raise ValueError("Cannot split a non-mixed element")

        return tuple([type(self)(subelem.element) for subelem in self._elem.elements])

    @property
    def _is_tensor_element(self):
        return isinstance(self._elem, finat.TensorFiniteElement)


def _wrapper_kernel_cache_key(form, split_knl, all_integer_subdomain_ids, **kwargs):
    if isinstance(form, ufl.Form):
        sig = form.signature()
    elif isinstance(form, slate.TensorBase):
        sig = form.expression_hash

    # The form signature does not store this information. This should be accessible from
    # the UFL so we don't need this nasty hack.
    subdomain_key = []
    for val in form.subdomain_data().values():
        for k, v in val.items():
            if v is not None:
                extruded = v._extruded
                constant_layers = extruded and v.constant_layers
                subset = isinstance(v, op2.Subset)
                subdomain_key.append((k, extruded, constant_layers, subset))
            else:
                subdomain_key.append((k,))

    return ((sig,)
            + tuple(subdomain_key)
            + tuplify(all_integer_subdomain_ids)
            + cachetools.keys.hashkey(split_knl, **kwargs))


@cachetools.cached(cachetools.LRUCache(maxsize=128), key=_wrapper_kernel_cache_key)
def _make_wrapper_kernel(*args, **kwargs):
    return _AssembleWrapperKernelBuilder(*args, **kwargs).build()


class ParloopExecutor:

    def __init__(self, form, split_knl, knl, all_integer_subdomain_ids, *, tensor=None, diagonal=False, lgmaps=None):
        """

        .. note::

            Here expr is a 'Firedrake-level' entity since we now recognise that data is
            attached. This means that we cannot safely cache the resulting object.
        """
        self._form = form
        self._split_knl = split_knl
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._kinfo = split_knl.kinfo
        self._knl = knl
        self._tensor = tensor
        self._diagonal = diagonal
        self._lgmaps = lgmaps

        self._iterset = self._get_iterset()

    def run(self):
        kinfo = self._split_knl.kinfo

        # Icky generator so we can access the correct coefficients in order
        def coeffs():
            for n, split_map in kinfo.coefficient_map:
                c = self._form.coefficients()[n]
                split_c = c.split()
                for c_ in (split_c[i] for i in split_map):
                    yield c_
        self.coeffs_iterator = iter(coeffs())

        parloop_args = [
            _as_parloop_arg(tsfc_arg, self)
            for tsfc_arg in self._split_knl.kinfo.tsfc_kernel_args if tsfc_arg.intent is not None
        ]
        try:
            op2.parloop(self._knl, self._iterset, parloop_args)
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all "
                               "coefficients/arguments")

    @functools.cached_property
    def mesh(self):
        return self._form.ufl_domains()[self._kinfo.domain_number]

    @property
    def integral_type(self):
        return self._kinfo.integral_type

    def _get_map(self, V):
        """TODO"""
        assert isinstance(V, ufl.FunctionSpace)

        if self.integral_type in {"cell", "exterior_facet_top",
                                  "exterior_facet_bottom",
                                  "interior_facet_horiz"}:
            return V.cell_node_map()
        elif self.integral_type in {"exterior_facet", "exterior_facet_vert"}:
            return V.exterior_facet_node_map()
        elif self.integral_type in {"interior_facet", "interior_facet_vert"}:
            return V.interior_facet_node_map()
        else:
            raise AssertionError

    def _get_iterset(self):
        mesh = self._form.ufl_domains()[self._kinfo.domain_number]
        subdomain_data = self._form.subdomain_data()[mesh].get(self._kinfo.integral_type, None)
        if subdomain_data is not None:
            if self._kinfo.integral_type != "cell":
                raise NotImplementedError("subdomain_data only supported with cell integrals")
            if self._kinfo.subdomain_id not in ["everywhere", "otherwise"]:
                raise ValueError("Cannot use subdomain data and subdomain_id")
            return subdomain_data
        else:
            return mesh.measure_set(self._kinfo.integral_type, self._kinfo.subdomain_id,
                                    self._all_integer_subdomain_ids)

    def rank1stuff(self, dat, V):
        if V.ufl_element().family() == "Real":
            return op2.GlobalParloopArg(dat)
        else:
            return op2.DatParloopArg(dat, self._get_map(V))


    def rank2stuff(self, tensor, Vrow, Vcol):
        if Vrow.ufl_element().family() == "Real":
            if Vcol.ufl_element().family() == "Real":
                return op2.GlobalParloopArg(tensor.handle.getPythonContext().global_)
            else:
                mp = self._get_map(Vcol)
                return op2.DatParloopArg(tensor.handle.getPythonContext().dat, mp)
        else:
            if Vcol.ufl_element().family() == "Real":
                mp = self._get_map(Vrow)
                return op2.DatParloopArg(tensor.handle.getPythonContext().dat, mp)
            else:
                rmap = self._get_map(Vrow)
                cmap = self._get_map(Vcol)
                return op2.MatParloopArg(tensor, (rmap, cmap), lgmaps=(self._lgmaps,))




# TODO Make into a singledispatchmethod when we have Python 3.8
@functools.singledispatch
def _as_parloop_arg(tsfc_arg, self):
    """Return a :class:`op2.ParloopArg` corresponding to the provided
    :class:`tsfc.KernelArg`.
    """
    raise NotImplementedError


@_as_parloop_arg.register(kernel_args.OutputKernelArg)
def _as_parloop_arg_output(_, self):
    arguments = self._form.arguments()

    if len(arguments) == 0:
        return op2.GlobalParloopArg(self._tensor)

    if self._diagonal:
        test, _ = arguments
        arguments = test,

    indices = self._split_knl.indices

    if any(i is not None for i in indices):
        assert all(i is not None for i in indices)

        func_spaces = [a.ufl_function_space()[idx] for idx, a in zip(indices, arguments)]

        if len(arguments) == 1:
            i, = indices
            V, = func_spaces
            return self.rank1stuff(self._tensor.dat[i], V)
        elif len(arguments) == 2:
            i, j = indices
            return self.rank2stuff(self._tensor.M[i, j], *func_spaces)
        else:
            raise AssertionError
    else:
        func_spaces = [a.ufl_function_space() for a in arguments]

        if len(arguments) == 1:
            V, = func_spaces
            return self.rank1stuff(self._tensor, V)
        elif len(arguments) == 2:
            return self.rank2stuff(self._tensor.M, *func_spaces)
        else:
            raise AssertionError


@_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
def _as_parloop_arg_coordinates(_, self):
    func = self.mesh.coordinates
    map_ = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.ConstantKernelArg)
def _as_parloop_arg_constant(_, self):
    coeff = next(self.coeffs_iterator)
    return op2.GlobalParloopArg(coeff.dat)


@_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
def _as_parloop_arg_coefficient(_, self):
    coeff = next(self.coeffs_iterator)
    mp = self._get_map(coeff.function_space())
    return op2.DatParloopArg(coeff.dat, mp)


@_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_parloop_arg_cell_orientations(_, self):
    func = self.mesh.cell_orientations()
    mp = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, mp)


@_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
def _as_parloop_arg_cell_sizes(_, self):
    func = self.mesh.cell_sizes
    mp = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, mp)


@_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_parloop_arg_exterior_facet(_, self):
    return op2.DatParloopArg(self.mesh.exterior_facets.local_facet_dat)


@_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_parloop_arg_interior_facet(_, self):
    return op2.DatParloopArg(self.mesh.interior_facets.local_facet_dat)


@_as_parloop_arg.register(CellFacetKernelArg)
def _as_parloop_arg_cell_facet(_, self):
    return op2.DatParloopArg(self.mesh.cell_to_facets)


@_as_parloop_arg.register(LayerCountKernelArg)
def _as_parloop_arg_layer_count(_, self):
    glob = op2.Global(LayerCountKernelArg.shape, self._iterset.layers-2,
                      dtype=LayerCountKernelArg.dtype)
    return op2.GlobalParloopArg(glob)


def _execute_parloop(*args, **kwargs):
    ParloopExecutor(*args, **kwargs).run()
