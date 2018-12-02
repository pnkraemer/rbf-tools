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

def buildKernMtrxMatern(firstPtset, secondPtset, smoothnessPar, lengthscalePar = 1.0, scalingPar = 1.0):
	def matern(pt1, pt2, reg = smoothnessPar):
		varDst = normDist(pt1, pt2)
		return maternfct(varDst, reg)
	mtrx = buildKernMtrx(firstPtset, secondPtset, matern)
	return mtrx

def buildKernMtrx(firstPtset, secondPtset, kernFct):
	lenFirstPts = len(firstPtset)
	lenSecPts = len(secondPtset)
	kernelMtrxFrame = np.zeros((lenFirstPts,lenSecPts))
	for rowIdx in range(lenFirstPts):
		for colIdx in range(lenSecPts):
			kernelMtrxFrame[rowIdx,colIdx] = kernFct(firstPtset[rowIdx,:], secondPtset[colIdx,:])
	return kernelMtrxFrame

def buildRhs(ptSet, fct):
	lenPtSet = len(ptSet)
	vecRhs = np.zeros(lenPtSet)
	for idx in range(lenPtSet):
		vecRhs[idx] = fxt(ptSet[idx,:])
	return vecRhs

def normDist(x, y, whichNorm = None):	# None is Frobenius
	return np.linalg.norm(x-y, ord = whichNorm)

def maternfct(distVar, smoothnessPar, lengthscalePar = 1.0, scalingPar = 1.0):
	if distVar <= 0:
		return scalingPar**2
	else:
 		scaledDistVar = np.sqrt(2*smoothnessPar)*distVar / lengthscalePar
		return scalingPar**2 * 2**(1-smoothnessPar) / scipy.special.gamma(smoothnessPar) \
			* scaledDistVar**(smoothnessPar) * scipy.special.kv(smoothnessPar, scaledDistVar)




