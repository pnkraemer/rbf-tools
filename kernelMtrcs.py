# NAME: 'kernelMtrcs.py'
#
# PURPOSE: Collection of scripts to construct kernel matrices
#
# DESCRIPTION: see PURPOSE
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
np.random.seed(15051994)
np.set_printoptions(precision = 2)

def buildKernelMtrxShift(ptSetOne, ptSetTwo, kernelFct, shiftPar):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	if lenPtSetOne != lenPtSetTwo:
		print "The pointsets do not align... return 0"
		return 0
	kernelMtrx = np.zeros((lenPtSetOne, lenPtSetTwo))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
				kernelMtrx[idx,jdx] = kernelFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
	return kernelMtrx + shiftPar * np.identity(lenPtSetOne)

def buildKernelMtrx(ptSetOne, ptSetTwo, kernelFct):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	kernelMtrx = np.zeros((lenPtSetOne, lenPtSetTwo))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			kernelMtrx[idx,jdx] = kernelFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
	return kernelMtrx

def buildKernelMtrxCond(ptSetOne, ptSetTwo, kernelFct, polOrder = 1):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	dim = ptSetOne.shape[1]
	kernelMtrx = np.zeros((lenPtSetOne + dim *polOrder + 1, lenPtSetTwo + dim *polOrder + 1))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			kernelMtrx[idx,jdx] = kernelFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
		for jdx in range(polOrder):
			kernelMtrx[idx, lenPtSetTwo] = 1
			for kdx in range(dim):
				kernelMtrx[idx, lenPtSetTwo + dim*jdx + kdx + 1] = ptSetOne[idx,kdx]**(jdx+1)
	for idx in range(lenPtSetTwo):
		kernelMtrx[lenPtSetOne, idx] = 1
		for jdx in range(polOrder):
			for kdx in range(dim):
				kernelMtrx[lenPtSetOne + dim*jdx + kdx + 1, idx] = ptSetTwo[idx,kdx]**(jdx+1)
	return kernelMtrx

# MV = matrix valued 
def buildKernelMtrxMV(ptSetOne, ptSetTwo, kernelFctMV, imageDim = 2):
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	kernelMtrxMV = np.zeros((imageDim*lenPtSetOne, imageDim*lenPtSetTwo))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			kernelMtrxMV[(imageDim*idx):(imageDim*(idx+1)), (imageDim*jdx):(imageDim*(jdx+1))] = kernelFctMV(ptSetOne[idx,:], ptSetTwo[jdx,:])
	return kernelMtrxMV

