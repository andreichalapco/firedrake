from abc import ABCMeta, abstractmethod
import numpy as np

import firedrake.cython.dmcommon as dmcommon
import firedrake.function as func
import firedrake.functionspace as fs
import firedrake.mesh as fmesh
import firedrake.utils as utils
import ufl


__all__ = ["RiemannianMetric", "IsotropicRiemannianMetric", "UniformRiemannianMetric",
           "MetricBasedAdaptor", "adapt"]


class Metric(object):
    """
    Abstract class that defines the API for all metrics.
    """
    __metaclass__ = ABCMeta
    metric_parameters = {}

    def __init__(self, mesh):
        """
        :arg mesh: mesh upon which to build the metric
        """
        self.mesh = mesh
        mesh.init()
        self.dim = mesh.topological_dimension()
        if self.dim not in (2, 3):
            raise ValueError(f"Mesh must be 2D or 3D, not {self.dim}")
        self.update_plex_coordinates()

    # TODO: This will be redundant at some point
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        plex = self.mesh.topology_dm
        entity_dofs = np.zeros(self.dim+1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coord_section = dmcommon.create_section(self.mesh, entity_dofs)
        self._coord_dm = plex.getCoordinateDM()
        self._coord_dm.setDefaultSection(coord_section)
        coords_local = self._coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(self.mesh.coordinates.dat.data_ro_with_halos,
                                           coords_local.array.shape)
        plex.setCoordinatesLocal(coords_local)

    @abstractmethod
    def set_parameters(self):
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
    """
    # TODO: Point to the section of the PETSc manual,
    #       once it has been updated.
    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        super().__init__(mesh)
        self.V = fs.TensorFunctionSpace(mesh, 'CG', 1)
        self.function = func.Function(self.V)

        self.plex = self._coord_dm.clone()
        entity_dofs = np.zeros(self.dim+1, dtype=np.int32)
        entity_dofs[0] = self.dim**2
        metric_section = dmcommon.create_section(self.mesh, entity_dofs)
        self.plex.setDefaultSection(metric_section)
        self._vec = self.plex.createLocalVec()

        self.set_parameters(metric_parameters)

    def _to_vec(self, function):
        return np.reshape(
            function.dat.data_ro_with_halos,
            self._vec.array.shape,
        )

    def _to_function(self, vec):
        return np.reshape(
            vec.array,
            self.function.dat.data_ro_with_halos.shape,
        )

    @property
    def vec(self):
        self._vec.array[:] = self._to_vec(self.function)
        return self._vec

    @property
    def dat(self):
        return self.function.dat

    @property
    def reordered(self):
        return dmcommon.to_petsc_local_numbering(self.vec, self.V)

    def process_parameters(self):
        """
        Process :attr:`metric_parameters` so that they can be
        used to drive metric-based mesh adaptation.
        """
        mp = self.metric_parameters.copy()
        if 'dm_plex_metric' in mp:
            for key, value in mp['dm_plex_metric'].items():
                mp['_'.join(['dm_plex_metric', key])] = value
            mp.pop('dm_plex_metric')
        mp.setdefault('dm_plex_metric_h_min', 1.0e-30)
        mp.setdefault('dm_plex_metric_h_max', 1.0e+30)
        mp.setdefault('dm_plex_metric_a_max', 1.0e+10)
        mp.setdefault('dm_plex_metric_normalization_order', 1.0)
        mp.setdefault('dm_plex_metric_gradation_factor', 1.3)
        self.metric_parameters = mp

    def set_parameters(self, metric_parameters={}):
        """
        Apply the :attr:`metric_parameters` to the DMPlex.
        """
        self.metric_parameters.update(metric_parameters)
        self.process_parameters()
        mp = self.metric_parameters
        # TODO: use metricSetFromOptions
        self.plex.metricSetMinimumMagnitude(mp['dm_plex_metric_h_min'])
        self.plex.metricSetMaximumMagnitude(mp['dm_plex_metric_h_max'])
        self.plex.metricSetMaximumAnisotropy(mp['dm_plex_metric_a_max'])
        self.plex.metricSetNormalizationOrder(mp['dm_plex_metric_normalization_order'])

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
        self.function.dat.data_with_halos[:] = self._to_function(tmp)

    def normalise(self, target_complexity=None, restrict_sizes=False, restrict_anisotropy=False):
        """
        Normalise the metric with respect to the provided target complexity.

        :kwarg target_complexity: analogous to desired mesh vertex count.
        :kwarg restrict_sizes: should minimum and maximum metric magnitudes
            be enforced?
        :kwarg restrict_anisotropy: should maximum anisotropy be enforced?
        """
        target = target_complexity or self.metric_parameters.get('dm_plex_metric_target_complexity')
        if target is None:
            raise ValueError('Cannot normalise a metric without a target complexity value.')
        self.plex.metricSetTargetComplexity(target)
        self._vec.copy(self.plex.metricNormalize(
            self.vec, restrictSizes=restrict_sizes, restrictAnisotropy=restrict_anisotropy,
        ))

    def intersect(self, *metrics):
        """
        Intersect the metric with other metrics.

        Metric intersection means taking the minimal ellipsoid in the
        direction of each eigenvector at each point in the domain.

        :arg metrics: the metrics to be intersected with
        """
        for metric in metrics:
            assert isinstance(metric, RiemannianMetric)
        num_metrics = len(metrics) + 1
        if num_metrics == 2:
            tmp = self.plex.metricIntersection2(self.vec, metrics[0].vec)
        elif num_metrics == 3:
            tmp = self.plex.metricIntersection3(self.vec, metrics[0].vec, metrics[1].vec)
        else:
            raise NotImplementedError(f'Can only intersect 2 or 3 metrics, not {num_metrics}')
        self._vec.copy(tmp)

    def rename(self, name):
        self.function.rename(name)

    def assign(self, *args, **kwargs):
        self.function.assign(*args, **kwargs)
        return self

    def interpolate(self, *args, **kwargs):
        self.function.interpolate(*args, **kwargs)
        return self

    def project(self, *args, **kwargs):
        self.function.project(*args, **kwargs)
        return self


class IsotropicRiemannianMetric(RiemannianMetric):
    """
    A :class:`RiemannianMetric` whose values are scalings of
    the identity matrix.

    Such a metric should give rise to isotropic meshes.
    """
    def __init__(self, mesh, **kwargs):
        """
        :arg mesh: mesh upon which to build the metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        super().__init__(mesh, **kwargs)
        self.id = ufl.Identity(self.dim)

    def assign(self, scaling, **kwargs):
        return super().assign(scaling*self.id, **kwargs)

    def interpolate(self, scaling, **kwargs):
        return super().interpolate(scaling*self.id, **kwargs)

    def project(self, scaling, **kwargs):
        return super().project(scaling*self.id, **kwargs)


class UniformRiemannianMetric(IsotropicRiemannianMetric):
    """
    A :class:`RiemannianMetric` that takes a constant
    values across the domain, which is a scaling of
    the identity matrix.

    Such a metric should give rise to uniform, isotropic
    meshes.
    """
    def __init__(self, mesh, scaling, **kwargs):
        """
        :arg mesh: mesh upon which to build the metric
        :arg scaling: uniform scaling parameter for the
            metric
        :kwarg metric_parameters: PETSc parameters for
            metric construction
        """
        super().__init__(mesh, **kwargs)
        self.interpolate(scaling)


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
    def __init__(self, mesh, metric):
        """
        :arg mesh: :class:`MeshGeometry` to be adapted.
        :arg metric: Riemannian metric :class:`Function`.
        """
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        coord_fe = mesh.coordinates.ufl_element()
        if (coord_fe.family(), coord_fe.degree()) != ('Lagrange', 1):
            raise NotImplementedError(f"Mesh coordinates must be P1, not {coord_fe}")
        assert isinstance(metric, RiemannianMetric)
        super().__init__(mesh)
        self.metric = metric

    @utils.cached_property
    def adapted_mesh(self):
        """
        Adapt the mesh with respect to the provided metric.

        :return: a new :class:`MeshGeometry`.
        """
        plex = self.mesh.topology_dm
        self.metric.enforce_spd(restrict_sizes=True, restrict_anisotropy=True)
        metric = self.metric.reordered
        newplex = plex.adaptMetric(metric, "Face Sets", "Cell Sets")
        return fmesh.Mesh(newplex)

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
