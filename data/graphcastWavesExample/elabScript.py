import xarray as xr

# the files wind_g_glb040w.nc and ww3.200912.nc were downloaded from https://jeodpp.jrc.ec.europa.eu/ftp/private/zyWV2md3r/zUQkJSOL5g1DeoCK/
dsCfsr = xr.open_dataset("wind_g_glb040w.nc")
dsCfsr200912 = ds.sel(time=slice("2009-12-01", "2010-01-01"))

dsWve0 = xr.open_dataset("ww3.200912.nc")
dsWve = dsWve0.interp(time=dsCfsr200912.time)

dsWnd = dsCfsr200912.interp(lon=dsWve.longitude, lat=dsWve.latitude)

hs = dsWve.hs
u10 = dsWnd.u10
v10 = dsWnd.v10
ds = xr.merge([u10, v10, hs])

ds.to_netcdf("waves200912.nc")

ds = ds.sel(time=slice('2009-12-01', "2009-12-08T19:00:00"))
ds = ds.expand_dims('batch')
ds = ds.expand_dims('level', axis=2)
ds['level'] = (["level"], [0]) 
ds["datetime"] = ds.time.expand_dims('batch')
ds = ds.drop_vars(["lon", "lat"])
ds = ds.rename({"longitude": "lon", "latitude": "lat"})

ds.to_netcdf("waves200912_1batch.nc")

dsstat = xr.Dataset(
          data_vars=dict(
                  hs=(["level"], [2])
                 ),
          coords=dict(
                  level=(["level"], [0])
                 )
               )

dsstat.to_netcdf("mean_by_level.nc")

dsstat = xr.Dataset(
          data_vars=dict(
                  hs=(["level"], [1])
                 ),
          coords=dict(
                  level=(["level"], [0])
                 )
               )
dsstat.to_netcdf("stddev_by_level.nc")

dsstat = xr.Dataset(
          data_vars=dict(
                  hs=(["level"], [.1])
                 ),
          coords=dict(
                  level=(["level"], [0])
                 )
               )
dsstat.to_netcdf("diffs_stddev_by_level.nc")



