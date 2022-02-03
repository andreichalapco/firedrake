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


def test_size_restriction(dim):
    """
    Test that enforcing a minimum magnitude
    larger than the domain means that there
    are as few elements as possible.
    """
    Id = Identity(dim)
    mesh = uniform_mesh(dim)
    mp = {'dm_plex_metric_h_min': 2.0}
    metric = RiemannianMetric(mesh, metric_parameters=mp)
    metric.interpolate(100.0*Id)
    metric.enforce_spd(restrict_sizes=True)
    expected = RiemannianMetric(mesh)
    expected.interpolate(0.25*Id)
    # assert np.isclose(errornorm(metric.function, expected.function), 0.0)
    try:
        newmesh = adapt(mesh, metric)
    except PETSc.Error as exc:
        if exc.ierr == 63:
            pytest.xfail("No mesh adaptation tools are installed")
        else:
            raise Exception(f"PETSc error code {exc.ierr}")
    assert newmesh.num_cells() < mesh.num_cells()
