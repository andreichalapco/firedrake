"""Tests for mixed Helmholtz convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *
from common import *


@pytest.mark.xfail(reason="Diagnosis or reason unknown")
@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("RT", 1, "DG", 0, "h", "DG", 0, (1, 4)), 0.9),
                          (("RT", 2, "DG", 0, "h", "DG", 1, (1, 4)), 1.9),
                          (("RT", 3, "DG", 0, "h", "DG", 2, (1, 4)), 2.9),
                          (("BDM", 1, "DG", 0, "h", "DG", 0, (1, 4)), 0.9),
                          (("DG", 0, "CG", 1, "v", "DG", 0, (1, 4)), 0.9),
                          (("DG", 0, "CG", 2, "v", "DG", 1, (1, 4)), 1.9)])
def test_scalar_convergence(testcase, convrate):
    hfamily, hdegree, vfamily, vdegree, ori, altfamily, altdegree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        mesh = extmesh(2**ii, 2**ii, 2**ii)

        horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HDiv(OuterProductElement(horiz_elt, vert_elt))
        V1 = FunctionSpace(mesh, product_elt)

        if ori == "h":
            # use same vertical variation, but different horizontal
            # (think about product of complexes...)
            horiz_elt = FiniteElement(altfamily, "triangle", altdegree)
            vert_elt = FiniteElement(vfamily, "interval", vdegree)
        elif ori == "v":
            # opposite
            horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
            vert_elt = FiniteElement(altfamily, "interval", altdegree)
        product_elt = OuterProductElement(horiz_elt, vert_elt)
        V2 = FunctionSpace(mesh, product_elt)

        f = Function(V2)
        exact = Function(V2)
        if ori == "h":
            f.interpolate(Expression("(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
            exact.interpolate(Expression("sin(x[0]*pi*2)*sin(x[1]*pi*2)"))
        elif ori == "v":
            f.interpolate(Expression("(1+4*pi*pi)*sin(x[2]*pi*2)"))
            exact.interpolate(Expression("sin(x[2]*pi*2)"))

        W = V1 * V2
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        a = (p*q - q*div(u) + dot(v, u) + div(v)*p)*dx
        L = f*q*dx

        out = Function(W)
        solve(a == L, out, solver_parameters={'ksp_type': 'gmres'})
        l2err[ii - start] = sqrt(assemble((out[2]-exact)*(out[2]-exact)*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
