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


from __future__ import division	#division of integers into decimal

import numpy as np
#import scipy.special
#import scipy.spatial



#from pointsets import *
#from rbfs import *


def buildKernMtrxMatern2(firstPtset, secondPtset, smoothnessPar, lengthscalePar = 1.0, scalingPar = 1.0, normOrd = None):
	# def maternkernel(pt1, pt2):
	# 	distVar = np.linalg.norm(pt1 - pt2, ord = normOrd)
	# 	if distVar <= 0:
	# 		return scalingPar**2
	# 	else:
 # 			scaledDistVar = np.sqrt(2*smoothnessPar)*distVar / lengthscalePar
	# 	return scalingPar**2 * 2**(1-smoothnessPar) / scipy.special.gamma(smoothnessPar) \
	# 		* scaledDistVar**(smoothnessPar) * scipy.special.kv(smoothnessPar, scaledDistVar)
	lenFirstPts = len(firstPtset)
	lenSecPts = len(secondPtset)
	kernelMtrxFrame = np.zeros((lenFirstPts,lenSecPts))
	for rowIdx in xrange(lenFirstPts):
		for colIdx in xrange(lenSecPts):
			kernelMtrxFrame[rowIdx,colIdx] = np.linalg.norm(firstPtset[rowIdx,:] - secondPtset[colIdx,:])
			# kernelMtrxFrame[rowIdx,colIdx] = maternkernel(firstPtset[rowIdx,:], secondPtset[colIdx,:])
			# print kernelMtrxFrame[rowIdx,colIdx]
	return kernelMtrxFrame



def rhsFct(var1):
	return var1**2.0 + 2*var1**3.1



dim = 2
print "\nHow many points per dimension for the interpolation?"
ptsPerDim = input("\tEnter: ")

print "\nWhich regularity for the Matern kernel?"
regularityPar = input("\tEnter: ")








ptsTotal = ptsPerDim**dim


ptSet = np.random.rand(ptsTotal, dim)
print ptSet




kernMtrx = buildKernMtrxMatern2(ptSet, ptSet, regularityPar)
print kernMtrx

# vecRhs = buildRhs(ptSet, rhsFct)

# solCoeff = np.linalg.solve(kernMtrx, vecRhs)

# evalPtSet = getptsRandomshiftlattice(ptsTotal, dim)
# evalKernMtrx = buildKernMtrxMatern(evalPtSet, ptSet, regularityPar)
# rbfApprx = evalKernMtrx.dot(solCoeff)

# trueEval = buildRhs(evalPtSet, rhsFct)

# apprxError = np.linalg.norm(trueEval - rbfApprx)


# print "\nError between function and RBF approx.: \n\terror =", apprxError
# print ""













