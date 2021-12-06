from firedrake import *
from petsc4py import PETSc
import pytest
import numpy as np
import sys


def uniform_mesh(dim, n=5):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D.")


def circle_in_square_mesh():
    from os.path import abspath, join, dirname
    cwd = abspath(dirname(__file__))
    return Mesh(join(cwd, "..", "meshes", "circle_in_square.msh"))


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_size_restriction(dim):
    """
    Test that enforcing a minimum magnitude
    larger than the domain means that there
    are as few elements as possible.
    """
    mesh = uniform_mesh(dim)
    mp = {'dm_plex_metric_h_min': 2.0}
    metric = UniformRiemannianMetric(mesh, 100.0, metric_parameters=mp)
    metric.enforce_spd(restrict_sizes=True)
    expected = UniformRiemannianMetric(mesh, 0.25)
    with expected.dat.vec_ro as v:
        assert np.allclose(metric.vec.array, v.array)
    newmesh = adapt(mesh, metric)
    num_cells = newmesh.num_cells()
    # expected = 2 if dim == 2 else 6  # pragmatic
    expected = 4 if dim == 2 else 38  # mmg
    assert num_cells == expected


@pytest.mark.parallel(nprocs=2)
def test_normalisation(dim):
    """
    Test that normalising a metric with
    respect to a given metric complexity and
    the normalisation order :math:`p=1` DTRT.
    """
    mesh = uniform_mesh(dim)
    target = 200.0 if dim == 2 else 2500.0
    mp = {
        'dm_plex_metric': {
            'target_complexity': target,
            'normalization_order': 1.0,
        }
    }
    metric = UniformRiemannianMetric(mesh, 100.0, metric_parameters=mp)
    try:
        metric.normalise()
    except PETSc.Error as exc:
        pytest.fail(f'FIXME: PETSc error code {exc.ierr}')  # FIXME
    expected = UniformRiemannianMetric(mesh, pow(target, 2.0/dim))
    metric._vec.axpy(-1, expected.vec)
    norm = metric._vec.norm()
    assert np.isclose(norm, 0.0)


@pytest.mark.parallel(nprocs=2)
def test_intersection(dim):
    """
    Test that intersecting two metrics gives
    results in a metric with the minimal ellipsoid.
    """
    mesh = uniform_mesh(dim)
    metric1 = UniformRiemannianMetric(mesh, 100.0)
    metric2 = UniformRiemannianMetric(mesh, 25.0)
    metric1.intersect(metric2)
    expected = UniformRiemannianMetric(mesh, 100.0)
    metric1._vec.axpy(-1, expected.vec)
    norm = metric1._vec.norm()
    assert np.isclose(norm, 0.0)


def test_preserve_cell_tags():
    """
    Test that any cell tags are preserved after
    mesh adaptation.
    """
    mesh = circle_in_square_mesh()
    metric = UniformRiemannianMetric(mesh, 100.0)
    newmesh = adapt(mesh, metric)

    tags = set(mesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    newtags = set(newmesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    assert tags == newtags, 'Cell tags do not match'
    assert tags == {3, 4}, 'Cell tags are not as expected'

    one = Constant(1.0)
    area = assemble(one*dx(3, domain=mesh))
    newarea = assemble(one*dx(3, domain=newmesh))
    assert np.isclose(area, newarea), 'Internal area not preserved'
    # FIXME: internal surfaces are *not quite* preserved


def test_preserve_facet_tags():
    """
    Test that any facet tags are preserved after
    mesh adaptation.
    """
    mesh = circle_in_square_mesh()
    metric = UniformRiemannianMetric(mesh, 100.0)
    newmesh = adapt(mesh, metric)
    newmesh.init()

    tags = set(mesh.exterior_facets.unique_markers)
    newtags = set(newmesh.exterior_facets.unique_markers)
    assert tags == newtags, 'Facet tags do not match'
    assert tags == {1, 2}, 'Facet tags are not as expected'

    one = Constant(1.0)
    bnd = assemble(one*ds(1, domain=mesh))
    newbnd = assemble(one*ds(1, domain=newmesh))
    assert np.isclose(bnd, newbnd), 'Boundary length not preserved'

    arc = assemble(one*ds(2, domain=mesh))
    newarc = assemble(one*ds(2, domain=newmesh))
    assert np.isclose(arc, newarc), 'Internal arc length not preserved'


if __name__ == "__main__":  # TODO: Drop this eventually
    """
    Test the metric-based mesh adaptation interface.

    The following command line parameters may be of
    interest:
        -dm_adaptor <pragmatic/mmg/parmmg>
        -dm_plex_metric_restrict_anisotropy_first

    as well as these parameters, which are specific
    to Mmg and ParMmg:
        -dm_plex_metric_no_insertion
        -dm_plex_metric_no_swapping
        -dm_plex_metric_no_movement
        -dm_plex_metric_gradation_factor <beta>
        -dm_plex_metric_verbosity <verbosity>
        -dm_plex_metric_num_iterations <niter>
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', help='Choose one of the tests')
    parser.add_argument('-dim', help='Spatial dimension')
    parser.add_argument('-target', help='Target metric complexity')
    parser.add_argument('-circle_in_square', help='Use special mesh')
    parsed_args = parser.parse_known_args()[0]
    d = int(parsed_args.dim or 2)
    if parsed_args.test is not None:
        if parsed_args.test == 'size_restriction':
            test_size_restriction(d)
        elif parsed_args.test == 'intersection':
            test_intersection(d)
        elif parsed_args.test == 'normalisation':
            test_normalisation(d)
        elif parsed_args.test == 'preserve_facet_tags':
            test_preserve_facet_tags()
        elif parsed_args.test == 'preserve_cell_tags':
            test_preserve_cell_tags()
        else:
            raise ValueError(f"Test {parsed_args.test} not recognised.")
        print(f'PASSED: test_{parsed_args.test}')
        sys.exit(0)
    cis = bool(parsed_args.circle_in_square or False)
    mp = {}
    if cis:
        assert d == 2
        mesh = circle_in_square_mesh()
    else:
        mesh = uniform_mesh(d)
    File(f'mesh{d}d.pvd').write(mesh.coordinates)
    metric = UniformRiemannianMetric(mesh, 100.0)
    if parsed_args.target is not None:
        metric.normalise(target_complexity=float(parsed_args.target))
    newmesh = adapt(mesh, metric)
    File(f'newmesh{d}d.pvd').write(newmesh.coordinates)
