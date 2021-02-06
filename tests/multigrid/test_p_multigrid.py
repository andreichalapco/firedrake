import pytest
from firedrake import *


@pytest.fixture(params=["triangles", "quadrilaterals"], scope="module")
def mesh(request):
    if request.param == "triangles":
        base = UnitSquareMesh(2, 2)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    elif request.param == "quadrilaterals":
        base = UnitSquareMesh(2, 2, quadrilateral=True)
        mh = MeshHierarchy(base, 1)
        mesh = mh[-1]
    return mesh


@pytest.fixture(params=["matfree", "aij"], scope="module")
def mat_type(request):
    return request.param


def test_p_multigrid_scalar(mesh, mat_type):
    V = FunctionSpace(mesh, "CG", 4)

    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_levels_transfer_mat_type": mat_type,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "mg",
          "pmg_mg_coarse_pc_mg_type": "multiplicative",
          "pmg_mg_coarse_mg_levels": relax,
          "pmg_mg_coarse_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_mg_coarse_pc_type": "gamg"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 5
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    assert ppc.getMGLevels() == 3
    assert ppc.getMGCoarseSolve().pc.getMGLevels() == 2


def test_p_multigrid_nonlinear_scalar(mesh, mat_type):
    V = FunctionSpace(mesh, "CG", 4)

    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner((Constant(1.0) + u**2) * grad(u), grad(v))*dx - inner(f, v)*dx

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "newtonls",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_levels_transfer_mat_type": mat_type,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "mg",
          "pmg_mg_coarse_pc_mg_type": "multiplicative",
          "pmg_mg_coarse_mg_levels": relax,
          "pmg_mg_coarse_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_mg_coarse_pc_type": "gamg"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.its <= 3


@pytest.mark.skipcomplex
def test_p_multigrid_vector():
    mesh = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(mesh, "CG", 4)
    u = Function(V)

    rho = Constant(2700)
    g = Constant(-9.81)
    B = Constant((0.0, rho*g))  # Body force per unit volume

    # Elasticity parameters
    E_, nu = 6.9e10, 0.334
    mu, lmbda = Constant(E_/(2*(1 + nu))), Constant(E_*nu/((1 + nu)*(1 - 2*nu)))

    # Linear elastic energy
    E = 0.5 * (
               2*mu * inner(sym(grad(u)), sym(grad(u)))*dx     # noqa: E126
               + lmbda * inner(div(u), div(u))*dx             # noqa: E126
               - inner(B, u)*dx                               # noqa: E126
    )                                                         # noqa: E126

    bcs = DirichletBC(V, zero((2,)), 1)

    F = derivative(E, u, TestFunction(V))
    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_rtol": 1.0e-8,
          "ksp_atol": 1.0e-8,
          "ksp_converged_reason": None,
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": "firedrake.PMGPC",
          "pmg_pc_mg_type": "full",
          "pmg_mg_levels_ksp_type": "chebyshev",
          "pmg_mg_levels_ksp_monitor_true_residual": None,
          "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
          "pmg_mg_levels_ksp_max_it": 2,
          "pmg_mg_levels_pc_type": "pbjacobi",
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "lu"}
    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 20
    assert solver.snes.ksp.pc.getPythonContext().ppc.getMGLevels() == 3


class MixedPMG(PMGPC):
    def coarsen_element(self, ele):
        return MixedElement([PMGPC.coarsen_element(self, sub) for sub in ele.sub_elements()])


@pytest.mark.skipcomplex
def test_p_multigrid_mixed():
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 4)
    Z = MixedFunctionSpace([V, V])

    z = Function(Z)
    E = 0.5 * inner(grad(z), grad(z))*dx - inner(Constant((1, 1)), z)*dx
    F = derivative(E, z, TestFunction(Z))

    bcs = [DirichletBC(Z.sub(0), 0, "on_boundary"),
           DirichletBC(Z.sub(1), 0, "on_boundary")]

    relax = {"ksp_type": "chebyshev",
             "ksp_monitor_true_residual": None,
             "ksp_norm_type": "unpreconditioned",
             "ksp_max_it": 3,
             "pc_type": "jacobi"}

    sp = {"snes_monitor": None,
          "snes_type": "ksponly",
          "ksp_type": "fgmres",
          "ksp_monitor_true_residual": None,
          "pc_type": "python",
          "pc_python_type": __name__ + ".MixedPMG",
          "mat_type": "aij",
          "pmg_pc_mg_type": "multiplicative",
          "pmg_mg_levels": relax,
          "pmg_mg_coarse_ksp_type": "richardson",
          "pmg_mg_coarse_ksp_max_it": 1,
          "pmg_mg_coarse_ksp_norm_type": "unpreconditioned",
          "pmg_mg_coarse_ksp_monitor": None,
          "pmg_mg_coarse_pc_type": "lu"}
    problem = NonlinearVariationalProblem(F, z, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()

    assert solver.snes.ksp.its <= 5
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    assert ppc.getMGLevels() == 3


def test_p_fas_scalar():
    mat_type = "matfree"
    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 4)

    # This problem is fabricated such that the exact solution
    # is resolved before reaching the finest level, hence no
    # work should be done in the finest level.
    # This will no longer be true for non-homogenous bcs, due
    # to the way firedrake imposes the bcs before injection.
    u = Function(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    f = x[0]*(1-x[0]) + x[1]*(1-x[1])
    bcs = DirichletBC(V, 0, "on_boundary")

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx

    # Due to the convoluted nature of the nested iteration
    # it is better to specify absolute tolerances only
    rhs = assemble(F, bcs=bcs)
    with rhs.dat.vec_ro as Fvec:
        Fnorm = Fvec.norm()

    rtol = 1E-8
    atol = rtol * Fnorm

    coarse = {
        "ksp_type": "preonly",
        "ksp_norm_type": None,
        "pc_type": "cholesky"}

    relax = {
        "ksp_type": "chebyshev",
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "ksp_max_it": 3,
        "pc_type": "jacobi"}

    pmg = {
        "snes_type": "ksponly",
        "ksp_atol": atol,
        "ksp_rtol": 1E-50,
        "ksp_type": "cg",
        "ksp_converged_reason": None,
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        "pmg_pc_mg_type": "multiplicative",
        "pmg_mg_levels": relax,
        "pmg_mg_levels_transfer_mat_type": mat_type,
        "pmg_mg_coarse": coarse}

    pfas = {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": atol,
        "snes_rtol": 1E-50,
        "snes_type": "python",
        "snes_python_type": "firedrake.PMGSNES",
        "pfas_snes_fas_type": "kaskade",
        "pfas_fas_levels": pmg,
        "pfas_fas_coarse": coarse}

    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=pfas)
    solver.solve()

    ppc = solver.snes.getPythonContext().ppc
    levels = ppc.getFASLevels()
    assert levels == 3
    assert ppc.getFASSmoother(levels-1).getLinearSolveIterations() == 0


def test_p_fas_nonlinear_scalar():
    mat_type = "matfree"
    N = 4
    dxq = dx(degree=3*N+2)  # here we also test coarsening of quadrature degree

    mesh = UnitSquareMesh(4, 4, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", N)
    u = Function(V)
    f = Constant(1)
    bcs = DirichletBC(V, 0, "on_boundary")

    # Regularized p-Laplacian
    p = 5
    eps = 1
    y = eps + inner(grad(u), grad(u))
    E = (1/p)*(y**(p/2))*dxq - inner(f, u)*dxq
    F = derivative(E, u, TestFunction(V))

    problem = NonlinearVariationalProblem(F, u, bcs)

    # Due to the convoluted nature of the nested iteration
    # it is better to specify absolute tolerances only
    rhs = assemble(F, bcs=bcs)
    with rhs.dat.vec_ro as Fvec:
        Fnorm = Fvec.norm()

    rtol = 1E-8
    atol = rtol * Fnorm

    newton = {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_type": "newtonls",
        "snes_max_it": 20,
        "snes_atol": atol,
        "snes_rtol": 1E-50}

    coarse = {
        "ksp_type": "preonly",
        "ksp_norm_type": None,
        "pc_type": "cholesky"}

    relax = {
        "ksp_type": "chebyshev",
        "ksp_norm_type": "unpreconditioned",
        "ksp_max_it": 3,
        "pc_type": "jacobi"}

    pmg = {
        "ksp_atol": atol*1E-1,
        "ksp_rtol": 1E-50,
        "ksp_type": "cg",
        "ksp_converged_reason": None,
        "ksp_monitor_true_residual": None,
        "ksp_norm_type": "unpreconditioned",
        "pc_type": "python",
        "pc_python_type": "firedrake.PMGPC",
        "pmg_pc_mg_type": "multiplicative",
        "pmg_mg_levels": relax,
        "pmg_mg_levels_transfer_mat_type": mat_type,
        "pmg_mg_coarse": coarse}

    npmg = {**newton, **pmg}
    ncrs = {**newton, **coarse}

    pfas = {
        "mat_type": "aij",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": atol,
        "snes_rtol": 1E-50,
        "snes_type": "python",
        "snes_python_type": "firedrake.PMGSNES",
        "pfas_snes_fas_type": "kaskade",
        "pfas_fas_levels": npmg,
        "pfas_fas_coarse": ncrs}

    def check_coarsen_quadrature(solver):
        Nq = set()
        level = solver._ctx
        while level is not None:
            p = level._problem
            for form in (p.F, p.J):
                Nq.update(set(f.metadata().get("quadrature_degree") for f in form.integrals()))
            level = level._coarse
        assert {3*1+2, 3*2+2, 3*4+2} <= Nq

    solver_pfas = NonlinearVariationalSolver(problem, solver_parameters=pfas)
    solver_pfas.solve()

    check_coarsen_quadrature(solver_pfas)
    ppc = solver_pfas.snes.getPythonContext().ppc
    levels = ppc.getFASLevels()
    iter_pfas = ppc.getFASSmoother(levels-1).getLinearSolveIterations()
    assert levels == 3

    u.interpolate(Constant(0))
    solver_npmg = NonlinearVariationalSolver(problem, solver_parameters=npmg)
    solver_npmg.solve()

    check_coarsen_quadrature(solver_npmg)
    iter_npmg = solver_npmg.snes.getLinearSolveIterations()
    assert 2*iter_pfas <= iter_npmg
