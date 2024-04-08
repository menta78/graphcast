import os
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import netCDF4



def plotMapAtTime(ncfilepath, timeIndex):
  ds = netCDF4.Dataset(ncfilepath)
 #hs = ds.variables['WWM_1'][timeIndex, :]
  hs = ds.variables['elev'][timeIndex, :]
 #hs = ds.variables['vwnd'][timeIndex, :]
  xs = ds.variables['x'][:]
  ys = ds.variables['y'][:]

  tm = ds.variables['time']
  dtm = netCDF4.num2date(tm[timeIndex], tm.units, 'standard')
  print('printing for date ' + str(dtm))
  
  f = plt.figure(figsize=[7, 6])
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines(resolution='10m')

  mx = np.percentile(hs.flatten(), 99.9)
  mn = np.percentile(hs.flatten(), .1)
 #mx = 0.1
 #mn = -0.2
 #mx = .3
 #hs[hs > mx - .1] = mx - .1
  levels = np.arange(mn, mx, .001)
  cmap = 'ocean_r'
  cmap = 'jet'
 #xs[xs < 0] = xs[xs < 0] + 360
  cf = plt.tricontourf(xs, ys, hs, levels, cmap=cmap)
  cf.cmap.set_over([.5,0,0])
  plt.triplot(xs, ys, linewidth=.1, color='k')
  lnd = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='lightgray')
  lndmsk = ax.add_feature(lnd)
  lndmsk.set_zorder(1)

 #cf.set_clim(0, 1.6)
  cb = plt.colorbar()
  cb.ax.set_ylabel('Elev (m)', fontsize=15)
  plt.title(str(dtm))
  f.tight_layout()
  plt.savefig('elev_t=' + str(timeIndex).zfill(3) + '.png', dpi=400)


if __name__ == '__main__':
  import pdb; pdb.set_trace()
  ncfilepath = '../mentaschi2023SshAdriaticSea_202201.nc'
 #ncfilepath = '/BGFS/DISASTER/mentalo/ClimateRuns/SCHISM/globalCoarse_unst/outputs_firstSuccRun/schout_24.nc'
  for timeIndex in range(32):
    plotMapAtTime(ncfilepath, timeIndex)
  plt.show()

