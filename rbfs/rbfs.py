############################################
# NAME:
# rbfs.py
#
# PURPOSE:
# Provide a collection of functions to assemble different 
# rbf-based kernel matrices
#
# DESCRIPTION:
# We include different norms, different radial basis functions
# and functions which construct kernel matrices
#
# AUTHOR: 
# Nicholas Kraemer, kraemer@ins.uni-bonn.de
#
############################################

from __future__ import division	#division of integers into decimal
import numpy as np 
import scipy.special
import scipy.spatial



def buildKernMtrxStd(firstPtset, secondPtset, kernFct):
	lenFirstPts = len(firstPtset)
	lenSecPts = len(secondPtset)
	kernelMtrxFrame = np.zeros((lenFirstPts,lenSecPts))
	for rowIdx in range(lenFirstPts):
		for colIdx in range(lenSecPts):
			kernelMtrxFrame[i,j] = kernFct(firstPtset[rowIdx,:], secondPtset[colIdx,:])
	return kernelMtrxFrame


def normDist(x, y, whichNorm = None):	# None is Frobenius
	return np.linalg.norm(x-y, ord = whichNorm)

def maternfct(distVar, smoothnessPar, lengthscalePar = 1.0, scalingPar = 1.0):
	if distVar <= 0:
		return scalingPar**2
	else:
 		scaledDistVar = np.sqrt(2*smoothnessPar)*distVar / lengthscalePar
		return scalingPar**2 * 2**(1-smoothnessPar) / scipy.special.gamma(smoothnessPar) \
			* scaledDistVar**(smoothnessPar) * scipy.special.kv(smoothnessPar, scaledDistVar)






