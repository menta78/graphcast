from typing import NamedTuple

import numpy as np

from alphaBetaLab import abTriangularMesh

from graphcast import model_utils


"""
TODO: FIND THE WAY TO SELECT A SUBSET OF NODES AND CREATE LOWER RESOLUTION "FACES" FOR CONNECTION BETWEEN FAR ELEMETS
"""
class __unstructuredMesh(NamedTuple):
    verticesLatLon: np.array
    vertices: np.array # these are in 3D cartesian coordinates
    faces: np.array


    def faces_to_edges(self):
        faces = self.faces
        assert faces.ndim == 2
        assert faces.shape[-1] == 3
        senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        return senders, receivers


def __createFromMsh(msh):
    vrtLatLon = []
    for pt in msh.nodes.values():
        vrtLatLon.append(pt[::-1])
    verticesLatLon = np.array(vrtLatLon)

    lat = verticesLatLon[:,0]
    lon = verticesLatLon[:,1]

    phi, theta = model_utils.lat_lon_deg_to_spherical(lat, lon)
    x, y, z = model_utils.spherical_to_cartesian(phi, theta)
    vertices = np.array([x, y, z]).transpose()

    faces = []
    for fc in msh.connectionPolygons.values():
        faces.append(np.array(fc)-1)
    faces = np.array(faces)
    return __unstructuredMesh(
            verticesLatLon=verticesLatLon,
            vertices=vertices,
            faces=faces
            )


def loadFromGr3(gr3FilePath: str):
    msh = abTriangularMesh.loadFromGr3File(gr3FilePath)
    return __createFromMsh(msh)


