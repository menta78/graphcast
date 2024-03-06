import numpy as np
import xarray as xr
from graphcast.icosahedral_mesh import TriangularMesh
from graphcast import model_utils


class MaskManager:

    def __init__(self):
        self.maskNcFilePath = "mask.nc"

    def initialize(self):
        ds = xr.open_dataset(self.maskNcFilePath)
        vls = ds.hs.values[0,0].squeeze()
        assert len(vls.shape) == 2
        self.gridMsk = np.isnan(vls)
        self.lon = ds.lon.values
        self.londiff = np.diff(self.lon)[0]
        self.lat = ds.lat.values
        self.latdiff = np.diff(self.lat)[0]
        ds.close()
        del ds

    def isMasked(self, lat, lon):
        ilon = np.floor( (lon - self.lon[0]) / self.londiff ).astype(int)
        ilat = np.floor( (lat - self.lat[0]) / self.latdiff ).astype(int)
        if (ilon >= len(self.lon)) or (ilat >= len(self.lat)):
            return True
        else:
            return self.gridMsk[ilat, ilon]

    def __pruneVerticesOnLandSingleMesh(self, mesh: TriangularMesh) -> TriangularMesh:
        faces = mesh.faces.tolist()
        vertices = []
        nvrt = len(mesh.vertices)

        def pruneFacesWithVertex(ivrt):
            nfaces = len(faces)
            invrng = range(nfaces-1,-1,-1)
            for iface in invrng:
                fci = faces[iface]
                if ivrt in fci:
                    faces.pop(iface)
                else:
                    for ivtfci in [0,1,2]:
                        if fci[ivtfci] > ivrt:
                            fci[ivtfci] -= 1

        for ivrt in range(nvrt-1,-1,-1):
            vrt3d = mesh.vertices[ivrt]
            sc1, sc2 = model_utils.cartesian_to_spherical(vrt3d[0], vrt3d[1], vrt3d[2])
            lat, lon = model_utils.spherical_to_lat_lon(sc1, sc2)
            if self.isMasked(lat, lon):
                pruneFacesWithVertex(ivrt)
            else:
                vertices.append(vrt3d)
        vertices = np.array(vertices)[::-1]
        faces = np.array(faces)
        mshout = TriangularMesh(
                vertices=vertices,
                faces=faces
                )
        return mshout

    def pruneMaskedIcosahedronVrtx(self, meshes: list) -> list:
        rslt = []
        for mesh in meshes:
            rslt.append(
                    self.__pruneVerticesOnLandSingleMesh(mesh))
        return rslt

    def pruneMaskedGridNodes(self, gridlats, gridlons) -> tuple:
        outgridlat, outgridlon = [], []
        for lat, lon in zip(gridlats, gridlons):
            if not self.isMasked(lat, lon):
                outgridlat.append(lat)
                outgridlon.append(lon)
        return np.array(outgridlat), np.array(outgridlon)








