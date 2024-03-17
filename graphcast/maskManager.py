import numpy as np
import xarray as xr
from jax import numpy as jnp
from graphcast.icosahedral_mesh import TriangularMesh
from graphcast import model_utils
from graphcast import xarray_jax


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

    def _getInvMaskAsXarrayDataset(self) -> xr.DataArray:
        msk = xr.DataArray(dims=('lat', 'lon'),
                           coords=dict(lat=self.lat, lon=self.lon),
                           data=~self.gridMsk)
        return msk.stack(dict(lat_lon_node=('lat', 'lon')))

    def maskFlatGriddedDataset(self, ds: xr.Dataset) -> xr.Dataset:
        dsgrp = ds.stack(dict(lat_lon_node=('lat', 'lon')))
        dimstrnsp = list(dsgrp.dims)
        dimstrnsp.remove('lat_lon_node')
        dimstrnsp.insert(0, 'lat_lon_node')
        dsgrptrnsp = dsgrp.transpose(*dimstrnsp)

        msk = self._getInvMaskAsXarrayDataset()

        dsmsktrnsp = dsgrptrnsp[msk]
        dims = list(dsgrp.dims)
        rslt = dsmsktrnsp.transpose(*dims)
        self.maskFlatLat = rslt.lat.values
        self.maskFlatLon = rslt.lon.values
        self.maskFlatLatLonNode = rslt.lat_lon_node.values
        return rslt

    def reCreateFlattenDataset(self, data):
        dims = ("lat_lon_node", "batch", "channels")
        return xarray_jax.DataArray(
            data=data,
            dims=dims,
            coords=dict(lat_lon_node=self.maskFlatLatLonNode))


    def maskUnflatGriddedDataset(self, vrbl: xr.DataArray) -> xr.DataArray:
        assert vrbl.dims.index('lat_lon_node') == 0, "maskUnflatGriddedDataset: lat_lon_node must be the 1st dimension"

        # getting the jax array of the input data
        dt = xarray_jax.unwrap(vrbl.data)

        # creating the jax array for the gridded results
        nlat = len(self.lat)
        nlon = len(self.lon)
        shp0 = list(vrbl.shape)
        shp = shp0.copy()
        shp[0] = nlat
        shp.insert(1, nlon)
        vls = jnp.zeros(shp, dtype=jnp.bfloat16)

        # broadcasting the mask to its shape
        expdim = [d for d in range(2, len(shp))]
        msk_ = np.expand_dims(~self.gridMsk, expdim)
        mskNd = np.broadcast_to(msk_, shp)

        # setting the values to the grid
        vls = vls.at[mskNd].set(dt.flatten())

        # creating the output dataset
        dims = list(vrbl.dims)
        dims.remove('lat_lon_node')
        dims.insert(0, 'lat')
        dims.insert(1, 'lon')
        vrunmsk = xarray_jax.DataArray(vls, dims=dims, coords=dict(lat=self.lat, lon=self.lon))
        return vrunmsk









