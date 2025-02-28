import numpy as np
import pytest
from firedrake import *


def test_constant():
    cg1 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 1)
    f = interpolate(Constant(1.0), cg1)
    assert np.allclose(1.0, f.dat.data)


def test_function():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    f = interpolate(x[0]*x[0], V1)
    g = interpolate(f, V2)

    # g shall be equivalent to:
    h = interpolate(x[0], V2)

    assert np.allclose(g.dat.data, h.dat.data)


def test_inner():
    m = UnitTriangleMesh()
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    x, y = SpatialCoordinate(m)
    f = interpolate(inner(x, x), V1)
    g = interpolate(f, V2)

    # g shall be equivalent to:
    h = interpolate(x, V2)

    assert np.allclose(g.dat.data, h.dat.data)


def test_coordinates():
    cg2 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 2)
    x = SpatialCoordinate(cg2.mesh())
    f = interpolate(x[0]*x[0], cg2)

    x = SpatialCoordinate(cg2.mesh())
    g = interpolate(x[0]*x[0], cg2)

    assert np.allclose(f.dat.data, g.dat.data)


def test_piola():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = interpolate(f[0], V)

    # g shall be equivalent to:
    h = project(f[0], V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_vector():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = VectorFunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_tensor():
    mesh = UnitSquareMesh(2, 2)
    x = SpatialCoordinate(mesh)
    U = TensorFunctionSpace(mesh, 'P', 1)
    V = TensorFunctionSpace(mesh, 'CG', 2)

    c = as_tensor(((Constant(2.0), x[1]), (x[0], x[0] * x[1])))

    f = project(c, U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_constant_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = interpolate(div(f), V)

    assert np.allclose(2.0, g.dat.data)


def test_compound_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 2)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = interpolate(Constant(1.5)*div(f) + sin(x[0] * np.pi), V)

    # g shall be equivalent to:
    h = interpolate(3.0 + sin(pi * x[0]), V)

    assert np.allclose(g.dat.data, h.dat.data)


# Requires the relevant FInAT or FIAT duals to be defined
@pytest.mark.xfail(raises=NotImplementedError, reason="Requires the relevant FInAT or FIAT duals to be defined")
def test_hdiv_2d():
    mesh = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    x = mesh.coordinates

    U = FunctionSpace(mesh, 'RTCF', 1)
    V = FunctionSpace(mesh, 'RTCF', 2)
    c = as_vector([x[1], -x[0], 0.0])

    f = project(c, U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


@pytest.mark.xfail(raises=NotImplementedError, reason="Requires the relevant FInAT or FIAT duals to be defined")
def test_hcurl_2d():
    mesh = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    x = mesh.coordinates

    U = FunctionSpace(mesh, 'RTCE', 1)
    V = FunctionSpace(mesh, 'RTCE', 2)
    c = as_vector([-x[1], x[0], 0.0])

    f = project(c, U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_cell_orientation():
    m = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)
    x = m.coordinates
    U = FunctionSpace(m, 'RTCF', 1)
    V = VectorFunctionSpace(m, 'DQ', 1)

    f = project(as_tensor([x[1], -x[0], 0.0]), U)
    g = interpolate(f, V)

    # g shall be close to:
    h = project(f, V)

    assert abs(g.dat.data - h.dat.data).max() < 1e-2


def test_cellvolume():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'DG', 0)

    f = interpolate(CellVolume(m), V)

    assert np.allclose(f.dat.data_ro, 0.125)


def test_cellvolume_higher_order_coords():
    m = UnitTriangleMesh()
    V = VectorFunctionSpace(m, 'P', 3)
    f = Function(V)
    f.interpolate(m.coordinates)

    # Warp mesh so that the bottom triangle line is:
    # x(x - 1)(x + a) with a = 19/12.0
    def warp(x):
        return x * (x - 1)*(x + 19/12.0)

    f.dat.data[1, 1] = warp(1.0/3.0)
    f.dat.data[2, 1] = warp(2.0/3.0)

    mesh = Mesh(f)
    g = interpolate(CellVolume(mesh), FunctionSpace(mesh, 'DG', 0))

    assert np.allclose(g.dat.data_ro, 0.5 - (1.0/4.0 - (1 - 19.0/12.0)/3.0 - 19/24.0))


def test_mixed():
    m = UnitTriangleMesh()
    x = m.coordinates
    V1 = FunctionSpace(m, 'BDFM', 2)
    V2 = VectorFunctionSpace(m, 'P', 2)
    f = Function(V1 * V2)
    f.sub(0).project(as_tensor([x[1], -x[0]]))
    f.sub(1).interpolate(as_tensor([x[0], x[1]]))

    V = FunctionSpace(m, 'P', 1)
    g = interpolate(dot(grad(f[0]), grad(f[3])), V)

    assert np.allclose(1.0, g.dat.data)


def test_lvalue_rvalue():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.assign(1.0)
    u.interpolate(u + 1.0)
    assert np.allclose(u.dat.data_ro, 2.0)


@pytest.mark.parametrize("degree", range(1, 4))
def test_interpolator_Pk(degree):
    mesh = UnitSquareMesh(10, 10, quadrilateral=False)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", degree)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_interpolator_spectral(degree):
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    x = SpatialCoordinate(mesh)
    fe1 = FiniteElement("CG", mesh.ufl_cell(), degree, variant="spectral")
    P1 = FunctionSpace(mesh, fe1)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_interpolator_equiquads(degree):
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    x = SpatialCoordinate(mesh)
    fe1 = FiniteElement("CG", mesh.ufl_cell(), degree, variant="equispaced")
    P1 = FunctionSpace(mesh, fe1)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_interpolator_equitris(degree):
    mesh = UnitSquareMesh(10, 10, quadrilateral=False)
    x = SpatialCoordinate(mesh)
    fe1 = FiniteElement("CG", mesh.ufl_cell(), degree, variant="equispaced")
    P1 = FunctionSpace(mesh, fe1)
    P2 = FunctionSpace(mesh, "CG", degree + 1)

    expr = x[0]**degree + x[1]**degree
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_interpolator_tets():
    mesh = UnitTetrahedronMesh()
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)

    expr = x[0] + x[1]
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_interpolator_extruded():
    mesh = ExtrudedMesh(UnitSquareMesh(10, 10), 10, 0.1)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)

    expr = x[0] + x[1]
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)

    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)


def test_trace():
    mesh = UnitSquareMesh(10, 10)
    x = SpatialCoordinate(mesh)
    cg = FunctionSpace(mesh, "CG", 1)
    tr = FunctionSpace(mesh, "HDiv Trace", 1)

    expr = x[0] + x[1]
    x_cg = interpolate(expr, cg)
    x_tr_dir = interpolate(expr, tr)
    x_tr_cg = interpolate(x_cg, tr)

    assert np.allclose(x_tr_cg.dat.data, x_tr_dir.dat.data)


@pytest.mark.parametrize("degree", range(1, 4))
def test_adjoint_Pk(degree):
    mesh = UnitSquareMesh(10, 10)
    Pkp1 = FunctionSpace(mesh, "CG", degree+1)
    Pk = FunctionSpace(mesh, "CG", degree)

    v = conj(TestFunction(Pkp1))
    u_Pk = assemble(conj(TestFunction(Pk)) * dx)
    interpolator = Interpolator(TestFunction(Pk), Pkp1)
    v_adj = interpolator.interpolate(assemble(v * dx), transpose=True)

    assert np.allclose(u_Pk.dat.data, v_adj.dat.data)


def test_adjoint_quads():
    mesh = UnitSquareMesh(10, 10)
    fe1 = FiniteElement("CG", mesh.ufl_cell(), 1, variant="equispaced")
    P1 = FunctionSpace(mesh, fe1)
    P2 = FunctionSpace(mesh, "CG", 2)

    v = conj(TestFunction(P2))
    u_P1 = assemble(conj(TestFunction(P1)) * dx)
    interpolator = Interpolator(TestFunction(P1), P2)
    v_adj = interpolator.interpolate(assemble(v * dx), transpose=True)

    assert np.allclose(u_P1.dat.data, v_adj.dat.data)


def test_adjoint_dg():
    mesh = UnitSquareMesh(10, 10)
    cg1 = FunctionSpace(mesh, "CG", 1)
    dg1 = FunctionSpace(mesh, "DG", 1)

    v = conj(TestFunction(dg1))
    u_cg = assemble(conj(TestFunction(cg1)) * dx)
    interpolator = Interpolator(TestFunction(cg1), dg1)
    v_adj = interpolator.interpolate(assemble(v * dx), transpose=True)

    assert np.allclose(u_cg.dat.data, v_adj.dat.data)


@pytest.mark.skipcomplex  # complex numbers are not orderable
@pytest.mark.parametrize("access", [MIN, MAX])
def test_interpolate_minmax(access):
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "DG", 0)
    x, y = SpatialCoordinate(mesh)
    g = interpolate(x*y, V)
    f = interpolate(x**2 - y**4, V)
    actual = Function(g)
    actual = interpolate(f, actual, access=access).dat.data_ro

    if access is MIN:
        expect = np.where(f.dat.data_ro < g.dat.data_ro, f.dat.data_ro, g.dat.data_ro)
    else:
        expect = np.where(f.dat.data_ro > g.dat.data_ro, f.dat.data_ro, g.dat.data_ro)

    assert np.allclose(actual, expect)


@pytest.mark.skipcomplex  # complex numbers are not orderable
def test_interpolate_periodic_coords_max():
    mesh = PeriodicUnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "P", 1)

    continuous = interpolate(SpatialCoordinate(mesh), V, access=MAX)

    # All nodes on the "seam" end up being 1, not 0.
    assert np.allclose(np.unique(continuous.dat.data_ro),
                       [0.25, 0.5, 0.75, 1])


def test_basic_dual_eval_cg3():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 3)
    x = SpatialCoordinate(mesh)
    expr = Constant(1.)
    f = interpolate(expr, V)
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values], [node(expr) for node in f.function_space().finat_element.fiat_equivalent.dual_basis()])
    expr = x[0]**3
    # Account for cell and corresponding expression being flipped onto
    # reference cell before reaching FIAT
    expr_fiat = (1-x[0])**3
    f = interpolate(expr, V)
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values], [node(expr_fiat) for node in f.function_space().finat_element.fiat_equivalent.dual_basis()])


def test_basic_dual_eval_bdm():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "BDM", 2)
    x = SpatialCoordinate(mesh)
    expr = as_vector([x[0], x[1]])
    f = interpolate(expr, V)
    dual_basis = f.function_space().finat_element.fiat_equivalent.dual_basis()
    # Can't do nodal evaluation of the FIAT dual basis yet so just check the
    # dat is the correct length
    assert len(f.dat.data_ro) == len(dual_basis)


def test_quadrature():
    from ufl.geometry import QuadratureWeight
    mesh = UnitIntervalMesh(1)
    Qse = FiniteElement("Quadrature", mesh.ufl_cell(), degree=2, quad_scheme="default")
    Qs = FunctionSpace(mesh, Qse)
    fiat_rule = Qs.finat_element.fiat_equivalent
    # For spatial coordinate we should get 2 points per cell
    x, = SpatialCoordinate(mesh)
    # Account for cell and corresponding expression being flipped onto
    # reference cell before reaching FIAT
    expr_fiat = 1-x
    xq = interpolate(expr_fiat, Qs)
    assert np.allclose(xq.dat.data_ro[xq.cell_node_map().values].T, fiat_rule._points)
    # For quadrature weight we should 2 equal weights for each cell
    w = QuadratureWeight(mesh)
    wq = interpolate(w, Qs)
    assert np.allclose(wq.dat.data_ro[wq.cell_node_map().values].T, fiat_rule._weights)


def test_interpolation_tensor_convergence():
    errors = []
    for n in range(2, 9):
        mesh = UnitSquareMesh(2**n, 2**n)
        # ||expr - I(expr)||_L2 = c h^k for degree k
        V = TensorFunctionSpace(mesh, "RT", 1)
        x, y = SpatialCoordinate(mesh)

        vs = V.ufl_element().value_shape()
        expr = as_tensor(np.asarray([
            sin(2*pi*x*(i+1))*cos(4*pi*y*i)
            for i in range(np.prod(vs, dtype=int))
        ], dtype=object).reshape(vs))

        f = interpolate(expr, V)

        errors.append(norm(expr - f))

    errors = np.asarray(errors)

    rate = np.log2(errors[:-1] / errors[1:])
    assert (rate[-2:] > 0.98).all()


def test_interpolation_tensor_symmetric():
    mesh = UnitSquareMesh(8, 7)
    # Interpolation of a symmetric tensor should be the same whether
    # we have symmetry or not.
    V = TensorFunctionSpace(mesh, "RT", 1, symmetry=True)
    Vexp = TensorFunctionSpace(mesh, "RT", 1)
    J = Jacobian(mesh)
    K = JacobianInverse(mesh)
    # Make a symmetric tensor-valued expression
    expr = as_tensor([J*J.T, K*K.T])
    expr = as_tensor(expr[i, j, k], (j, k, i))
    f = interpolate(expr, V)
    fexp = interpolate(expr, Vexp)
    assert np.isclose(norm(fexp - f), 0)
