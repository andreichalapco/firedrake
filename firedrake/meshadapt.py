from abc import ABCMeta, abstractmethod
import numpy as np

import firedrake.cython.dmcommon as dmcommon
import firedrake.function as func
import firedrake.functionspace as fs
import firedrake.mesh as fmesh
import firedrake.utils as utils
import ufl


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
