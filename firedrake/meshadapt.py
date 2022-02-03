from abc import ABCMeta, abstractmethod
import numpy as np

import firedrake.cython.dmcommon as dmcommon
import firedrake.function as func
import firedrake.functionspace as fs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
import firedrake.utils as utils
import ufl


__all__ = ["RiemannianMetric", "MetricBasedAdaptor", "adapt"]



class Metric(object):
    """
    Abstract class that defines the API for all metrics.
    """
    __metaclass__ = ABCMeta

    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        """
        mesh.init()
        self.dim = mesh.topological_dimension()
        if self.dim not in (2, 3):
            raise ValueError(f"Mesh must be 2D or 3D, not {self.dim}")
        self.mesh = mesh
        self.update_plex_coordinates()
        self.set_metric_parameters(metric_parameters)

    # TODO: This will be redundant at some point
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        self.plex = self.mesh.topology_dm
        entity_dofs = np.zeros(self.dim+1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coord_section = self.mesh.create_section(entity_dofs)
        # FIXME: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = self.plex.getCoordinateDM()
        coord_dm.setDefaultSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(self.mesh.coordinates.dat.data_ro_with_halos,
                                           coords_local.array.shape)
        self.plex.setCoordinatesLocal(coords_local)

    @abstractmethod
    def set_parameters(self, metric_parameters={}):
        """
        Set the :attr:`metric_parameters` so that they can be
        used to drive the mesh adaptation routine.
        """
        pass



class RiemannianMetric(Metric):
    r"""
    Class for defining a Riemannian metric over a
    given mesh.

    A metric is a symmetric positive-definite field,
    which conveys how the mesh is to be adapted. If
    the mesh is of dimension :math:`d` then the metric
    takes the value of a square :math:`d\times d`
    matrix at each point.

    The implementation of metric-based mesh adaptation
    used in PETSc assumes that the metric is piece-wise
    linear and continuous, with its degrees of freedom
    at the mesh vertices.

    For details, see the PETSc manual entry:
      https://petsc.org/release/docs/manual/dmplex.html#metric-based-mesh-adaptation
    """
    @PETSc.Log.EventDecorator("RiemannianMetric.__init__")
    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        super().__init__(mesh, metric_parameters=metric_parameters)
        self.V = fs.TensorFunctionSpace(mesh, 'CG', 1)
        self.function = func.Function(self.V)

    @property
    def dat(self):
        return self.function.dat

    @property
    def vec(self):
        with self.dat.vec_ro as v:
            return v

    @property
    @PETSc.Log.EventDecorator("RiemannianMetric.reordered")
    def reordered(self):
        return dmcommon.to_petsc_local_numbering(self.vec, self.V)

    @staticmethod
    def _process_parameters(metric_parameters):
        mp = metric_parameters.copy()
        if 'dm_plex_metric' in mp:
            for key, value in mp['dm_plex_metric'].items():
                mp['_'.join(['dm_plex_metric', key])] = value
            mp.pop('dm_plex_metric')
        return mp

    def set_metric_parameters(self, metric_parameters={}):
        """
        Apply the :attr:`metric_parameters` to the DMPlex.
        """
        mp = self._process_parameters(metric_parameters)
        OptDB = PETSc.Options()
        for key, value in mp.items():
            OptDB.setValue(key, value)
        self.plex.metricSetFromOptions()

    @PETSc.Log.EventDecorator("RiemannianMetric.enforce_spd")
    def enforce_spd(self, restrict_sizes=False, restrict_anisotropy=False):
        """
        Enforce that the metric is symmetric positive-definite.

        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        """
        tmp = self.plex.metricEnforceSPD(
            self.vec, restrictSizes=restrict_sizes, restrictAnisotropy=restrict_anisotropy,
        )
        with self.dat.vec_wo as v:
            v.copy(tmp)
        return self

    @PETSc.Log.EventDecorator("RiemannianMetric.normalise")
    def normalise(self, restrict_sizes=False, restrict_anisotropy=False):
        raise NotImplementedError  # TODO

    @PETSc.Log.EventDecorator("RiemannianMetric.intersect")
    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the
        direction of each eigenvector at each point in the domain.

        :arg metrics: the metrics to be intersected with
        """
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
        num_metrics = len(metrics)
        if num_metrics == 0:
            return self
        elif num_metrics == 1:
            tmp = self.plex.metricIntersection2(self.vec, metrics[0].vec)
        elif num_metrics == 2:
            tmp = self.plex.metricIntersection3(self.vec, metrics[0].vec, metrics[1].vec)
        else:
            raise NotImplementedError(f'Can only intersect 1, 2 or 3 metrics, not {num_metrics+1}')
        with self.dat.vec_wo as v:
            v.copy(tmp)
        return self

    def rename(self, name):
        self.function.rename(name)
        with self.dat.vec_wo as v:
            v.setName(name)

    def assign(self, *args, **kwargs):
        self.function.assign(*args, **kwargs)
        return self

    def interpolate(self, *args, **kwargs):
        self.function.interpolate(*args, **kwargs)
        return self

    def project(self, *args, **kwargs):
        self.function.project(*args, **kwargs)
        return self


class AdaptorBase(object):
    """
    Abstract class that defines the API for all mesh adaptors.
    """
    __metaclass__ = ABCMeta

    def __init__(self, mesh):
        """
        :arg mesh: mesh to be adapted
        """
        self.mesh = mesh

    @abstractmethod
    def adapted_mesh(self):
        pass

    @abstractmethod
    def interpolate(self, f):
        """
        Interpolate a field from the initial mesh to the adapted mesh.

        :arg f: the field to be interpolated
        """
        pass


class MetricBasedAdaptor(AdaptorBase):
    """
    Class for driving metric-based mesh adaptation.
    """
    @PETSc.Log.EventDecorator("MetricBasedAdaptor.__init__")
    def __init__(self, mesh, metric):
        """
        :arg mesh: :class:`MeshGeometry` to be adapted.
        :arg metric: Riemannian metric :class:`Function`.
        """
        if metric.mesh is not mesh:
            raise ValueError("The mesh associated with the metric is inconsistent")
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        coord_fe = mesh.coordinates.ufl_element()
        if (coord_fe.family(), coord_fe.degree()) != ('Lagrange', 1):
            raise NotImplementedError(f"Mesh coordinates must be P1, not {coord_fe}")
        assert isinstance(metric, RiemannianMetric)
        super().__init__(mesh)
        self.metric = metric

    @utils.cached_property
    @PETSc.Log.EventDecorator("MetricBasedAdaptor.adapted_mesh")
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :return: a new :class:`MeshGeometry`.
        """
        plex = self.mesh.topology_dm
        self.metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        metric = self.metric.reordered
        newplex = plex.adaptMetric(metric, "Face Sets", "Cell Sets")
        return fmesh.Mesh(newplex, distribution_parameters={"partition": False})

    @PETSc.Log.EventDecorator("MetricBasedAdaptor.interpolate")
    def interpolate(self, f):
        raise NotImplementedError  # TODO: Implement consistent interpolation in parallel


def adapt(mesh, *metrics, **kwargs):
    r"""
    Adapt a mesh with respect to a metric and some adaptor parameters.

    If multiple metrics are provided, then they are intersected.

    :arg mesh: :class:`MeshGeometry` to be adapted.
    :arg metrics: Riemannian metric :class:`Function`\s.
    :kwarg adaptor_parameters: parameters used to drive
        the metric-based mesh adaptation
    """
    num_metrics = len(metrics)
    metric = metrics[0]
    if num_metrics > 1:
        metric.intersect(*metrics[1:])
    adaptor = MetricBasedAdaptor(mesh, metric)
    return adaptor.adapted_mesh
