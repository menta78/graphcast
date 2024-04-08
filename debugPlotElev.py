import os
import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import netCDF4

from alphaBetaLab import abTriangularMesh



def plotMapAtTime(ds, timeIndex=0):
  hs = ds.elev.isel(dict(time=timeIndex)).values.squeeze()

  msh = abTriangularMesh.loadFromGr3File('./data/graphcastStormSurgeAdriaticSea/util/adriatic.gr3')
  xs = np.array([p[0] for p in msh.nodes.values()])
  ys = np.array([p[1] for p in msh.nodes.values()])

  tmstr = str(ds.time.values[0])
  
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
  plt.title(tmstr)
  f.tight_layout()
  plt.savefig('elev_t=' + tmstr + '.png', dpi=400)


if __name__ == '__main__':
  import pdb; pdb.set_trace()
  ncfilepath = '../mentaschi2023SshAdriaticSea_202201.nc'
 #ncfilepath = '/BGFS/DISASTER/mentalo/ClimateRuns/SCHISM/globalCoarse_unst/outputs_firstSuccRun/schout_24.nc'
  for timeIndex in range(32):
    plotMapAtTime(ncfilepath, timeIndex)
  plt.show()

