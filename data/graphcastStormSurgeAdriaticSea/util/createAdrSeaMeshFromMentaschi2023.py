from alphaBetaLab import abTriangularMesh


mshFlPth = "/mnt/c/Users/utente/Dropbox/università/schism/globalERA5Hindcast/setup/globalHR.gr3"
outMshFlPth = "./adriatic.gr3"

xsPolygon = [10.7639, 13.3567, 18.2785, 21.1350, 14.0158]
ysPolygon = [45.5381, 42.2784, 40.0624, 41.1632, 46.7860]

msh = abTriangularMesh.loadFromGr3File(mshFlPth)
clpmsh = msh.clipToPolygon(xsPolygon, ysPolygon)
clpmsh.saveAsGr3(outMshFlPth, bathyFactor=1)

