# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake.cython import dmplex
from firedrake.petsc import PETSc
from pyop2.datatypes import IntType
import ufl


@pytest.mark.parallel
@pytest.mark.parametrize("f_lambda", [lambda x: x[0] < 1.0001, lambda x: x[0] > 0.9999])
@pytest.mark.parametrize("b_lambda", [lambda x: x[0] > 0.9999, lambda x: x[0] < 1.0001, lambda x: x[1] < 0.0001, lambda x: x[1] > 0.9999])
def test_submesh_poisson_cell(f_lambda, b_lambda):

    # This test is for checking an edge case
    # where we have few elements.

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 111, "cell", None, geometric_expr = f_lambda)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", None, geometric_expr = b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    bc1 = DirichletBC(V, g, 1)

    parameters = {"mat_type": "aij",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)


@pytest.mark.parametrize("f_lambda", [lambda x: x[0] < 1.0001, lambda x: x[0] > 0.9999])
@pytest.mark.parametrize("b_lambda", [lambda x: x[0] > 0.9999, lambda x: x[0] < 1.0001, lambda x: x[1] < 0.0001, lambda x: x[1] > 0.9999])
def test_submesh_poisson_cell_error(f_lambda, b_lambda):

    msh = RectangleMesh(200, 100, 2., 1., quadrilateral=True)
    msh.init()

    x, y = SpatialCoordinate(msh)
    DP = FunctionSpace(msh, 'DP', 0)
    fltr = Function(DP)
    fltr = Function(DP).interpolate(ufl.conditional(real(x) < 1, 1, 0))

    #msh.markSubdomain("half_domain", 111, "cell", fltr, geometric_expr = f_lambda)
    msh.markSubdomain("half_domain", 111, "cell", fltr)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    # mesh_topology._facets (exterior_facets, interior_facets) is cached,
    # so dmplex.FACE_SETS_LABEL must be set before any call of _facets.
    # This makes it difficult to create a FunctionSpace here.
    # So for now we only allow lambda expression to set dmplex.FACE_SETS_LABEL.
    #RTCF = FunctionSpace(submsh, 'RTCF', 1)
    #fltr = Function(RTCF).project(as_vector([ufl.conditional(real(x) < 1, n * (x - 1), 0), 0]))
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", None, geometric_expr = b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    bc1 = DirichletBC(V, g, 1)

    parameters = {"mat_type": "aij",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)

    assert(sqrt(assemble(inner(u - g, u - g) * dx)) < 0.00016)


def test_submesh_helmholtz():

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 222, "cell", None, geometric_expr = lambda x: x[0] > 0.9999)

    submsh = SubMesh(msh, "half_domain", 222, "cell")

    V0 = FunctionSpace(msh, "CG", 1)
    V1 = FunctionSpace(submsh, "CG", 1)

    W = V0 * V1

    w = Function(W)
    u0, u1 = TrialFunctions(W)
    v0, v1 = TestFunctions(W)

    f0 = Function(V0)
    x0, y0 = SpatialCoordinate(msh)
    f0.interpolate(-8.0 * pi * pi * cos(x0 * pi * 2) * cos(y0 * pi * 2))

    dx0 = Measure("cell", domain=msh)
    dx1 = Measure("cell", domain=submsh)


    a = inner(grad(u0), grad(v0)) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(f0, v0) * dx0

    mat = assemble(a)
