from firedrake import *
from firedrake.meshadapt import *
from petsc4py import PETSc
import pytest
import numpy as np


def uniform_mesh(dim, n=5):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_intersection(dim):
    """
    Test that intersecting two metrics results
    in a metric with the minimal ellipsoid.
    """
    Id = Identity(dim)
    mesh = uniform_mesh(dim)
    metric1 = RiemannianMetric(mesh)
    metric1.interpolate(100.0*Id)
    metric2 = RiemannianMetric(mesh)
    metric2.interpolate(25.0*Id)
    metric1.intersect(metric2)
    expected = RiemannianMetric(mesh)
    expected.interpolate(100.0*Id)
    assert np.isclose(errornorm(metric1.function, expected.function), 0.0)
