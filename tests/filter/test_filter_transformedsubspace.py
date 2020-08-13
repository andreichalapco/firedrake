import pytest
from firedrake import *
from firedrake import ufl_expr
import numpy as np


def _rotate_mesh(mesh, theta):
    x, y = SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    coords = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                                 sin(theta) * x + cos(theta) * y]))
    return Mesh(coords)


def _poisson_analytical(V, xi, eta, which):
    if which == 'solution':
        #return sin(2 * pi * xi) * sin(2 * pi * eta)
        return cos(2 * pi * xi) * cos(2 * pi * eta)
    elif which == 'force':
        #return 8.0 * pi * pi * sin(2 * pi * xi) * sin(2 * pi * eta)
        return 8.0 * pi * pi * cos(2 * pi * xi) * cos(2 * pi * eta)


def _poisson_get_forms_original(V, f, n):
    u = TrialFunction(V)
    v = TestFunction(V)
    normal = FacetNormal(V.mesh())
    alpha = 100
    h = 1./(2**n)
    a = dot(grad(v), grad(u)) * dx - dot(grad(u), normal) * v * ds - dot(grad(v), normal) * u * ds + alpha / h * u * v * ds
    L = f * v * dx
    return a, L


def _poisson_get_forms_hermite(V, xprime, yprime, g, f):

    # Filter test function
    u = TrialFunction(V)
    v = TestFunction(V)

    V1 = BCSubspace(V, (1,2,3,4))

    ub = Masked(u, V1)
    vb = Masked(v, V1)

    gb = Masked(g, V1)

    # Make sure to project with very small tolerance.
    ud = Masked(u, V1.complement)
    vd = Masked(v, V1.complement)

    a = inner(grad(ud), grad(vd)) * dx + inner(ub, vb)* ds
    L = inner(f, vd) * dx - inner(grad(gb), grad(vd)) * dx + inner(gb, vb) * ds

    return a, L


def _poisson(n, el_type, degree, perturb):
    mesh = UnitSquareMesh(2**n, 2**n)
    if perturb:
        V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    # Rotate mesh (Currently only works with unrotated mesh)
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xprime = cos(theta) * x + sin(theta) * y
    yprime = -sin(theta) * x + cos(theta) * y

    # Define forms
    
    solver_parameters = {"ksp_rtol": 1.e-18}
    f = Function(V).project(_poisson_analytical(V, xprime, yprime, 'force'), solver_parameters=solver_parameters)
    g = Function(V).project(_poisson_analytical(V, xprime, yprime, 'solution'), solver_parameters=solver_parameters)
    #a, L = _poisson_get_forms_original(V, f, n)
    a, L = _poisson_get_forms_hermite(V, xprime, yprime, g, f)
    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'preonly', "pc_type": 'lu'})
    #solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'gmres',
    #                                              "ksp_rtol": 1.e-12,
    #                                              "ksp_atol": 1.e-12})

    # Postprocess
    sol_exact = _poisson_analytical(V, xprime, yprime, 'solution')
    sol_exact_proj = Function(V).project(_poisson_analytical(V, xprime, yprime, 'solution'), solver_parameters=solver_parameters)
    err = sqrt(assemble(dot(sol - sol_exact, sol - sol_exact) * dx))
    berr = sqrt(assemble(dot(sol - sol_exact_proj, sol - sol_exact_proj) * ds))
    print("error            : ", err)
    print("error on boundary: ", berr)
    """
    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True)
    #triangles = tripcolor(sol, axes=axes, cmap='coolwarm')
    triangles = tripcolor(Function(V).interpolate(sol_exact), axes=axes, cmap='coolwarm')
    axes.set_aspect("equal")
    axes.set_title("Pressure")
    fig.colorbar(triangles, ax=axes, fraction=0.032, pad=0.02)

    plt.savefig('temphermite.pdf')
    """
    return err, berr


def test_subspace_transformedsubspace_poisson_zany():
    """
    for el, deg, convrate in [('CG', 3, 4),
                              ('CG', 4, 5),
                              ('CG', 5, 6)]:
    for el, deg, convrate in [('Hermite', 3, 3.8),
                              ('Bell', 5, 4.8),
                              ('Argyris', 5, 4.8)]:
        diff = np.array([poisson(i, el, deg, True) for i in range(3, 8)])
        conv = np.log2(diff[:-1] / diff[1:])
        print(conv)
        #assert (np.array(conv) > convrate).all()
    """
    for el, deg, convrate in [('Hermite', 3, 4.0),]:
        errs = []
        for i in range(4, 9):
            err, berr = _poisson(i, el, deg, True)
            errs.append(err)
            assert(berr < 1e-12)
        errs = np.array(errs)
        conv = np.log2(errs[:-1] / errs[1:])
        print(conv)
        assert (np.array(conv) > convrate).all()
