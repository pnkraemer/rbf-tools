############################################
# NAME:
# interpolation2d.py

# PURPOSE:
# Provide a simple 2D rbf interpolation program

# DESCRIPTION:
# We construct an rbf interpolation to a 
# function and compute the error

# AUTHOR: 
# Nicholas Kraemer, kraemer@ins.uni-bonn.de
############################################




import pointsets.pointsets as pts
import rbfs.rbfs as rbfs
import numpy as np



def rhsFct(var1):
	return var1**2.0 + 2*var1**3.1






dim = 2
print "\nHow many points per dimension for the interpolation?"
ptsPerDim = input("\tEnter: ")

print "\nWhich regularity for the Matern kernel?"
regularityPar = input("\tEnter: ")








ptsTotal = ptsPerDim**dim


ptSet = pts.getptsTensorgrid(ptsTotal, dim)


kernMtrx = rbfs.buildKernMtrxMatern(ptSet, ptSet, regularityPar)


vecRhs = rbfs.buildRhs(ptSet, rhsFct)

solCoeff = np.linalg.solve(kernMtrx, vecRhs)

evalPtSet = pts.getptsRandomshiftlattice(ptsTotal, dim)
evalKernMtrx = rbfs.buildKernMtrxMatern(evalPtSet, ptSet, regularityPar)
rbfApprx = evalKernMtrx.dot(solCoeff)

trueEval = rbfs.buildRhs(evalPtSet, rhsFct)

apprxError = np.linalg.norm(trueEval - rbfApprx)


print "\nError between function and RBF approx.: \n\terror =", apprxError
print ""













