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
	dim = len(ptSetOne.T)
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

def buildKernelMtrxCondSph2(ptSetOne, ptSetTwo, kernelFct):
	def sph00(pt1, pt2, pt3):
		return 1
	def sph10(pt1, pt2, pt3):
		return pt1
	def sph11(pt1, pt2, pt3):
		return pt2
	def sph12(pt1, pt2, pt3):
		return pt3
	def sph20(pt1, pt2, pt3):
		return pt1*pt2
	def sph21(pt1, pt2, pt3):
		return pt2*pt3
	def sph22(pt1, pt2, pt3):
		return 2*pt3**2 - pt1**2 - pt2**2
	def sph23(pt1, pt2, pt3):
		return pt1*pt3
	def sph24(pt1, pt2, pt3):
		return pt1**2 - pt2**2
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	dim = len(ptSetOne.T)
	kernelMtrx = np.zeros((lenPtSetOne + 9, lenPtSetTwo + 9))
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			kernelMtrx[idx,jdx] = kernelFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
		kernelMtrx[idx, lenPtSetTwo] = sph00(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 1] = sph10(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 2] = sph11(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 3] = sph12(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 4] = sph20(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 5] = sph21(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 6] = sph22(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 7] = sph23(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 8] = sph24(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
	for idx in range(lenPtSetTwo):
		kernelMtrx[lenPtSetOne, idx] = sph00(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 1, idx] = sph10(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 2, idx] = sph11(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 3, idx] = sph12(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 4, idx] = sph20(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 5, idx] = sph21(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 6, idx] = sph22(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 7, idx] = sph23(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 8, idx] = sph24(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
	return kernelMtrx

def buildCollMtrxUnsymmCond(ptSetOne, ptSetTwo, diffKernelFct, pdeParam = 1.0):
	def sph00(pt1, pt2, pt3):
		return 1
	def sph10(pt1, pt2, pt3):
		return pt1
	def sph11(pt1, pt2, pt3):
		return pt2
	def sph12(pt1, pt2, pt3):
		return pt3
	def sph20(pt1, pt2, pt3):
		return pt1*pt2
	def sph21(pt1, pt2, pt3):
		return pt2*pt3
	def sph22(pt1, pt2, pt3):
		return 2*pt3**2 - pt1**2 - pt2**2
	def sph23(pt1, pt2, pt3):
		return pt1*pt3
	def sph24(pt1, pt2, pt3):
		return pt1**2 - pt2**2
	lenPtSetOne = len(ptSetOne)
	lenPtSetTwo = len(ptSetTwo)
	dim = len(ptSetOne.T)
	kernelMtrx = np.zeros((lenPtSetOne + 9, lenPtSetTwo + 9))
	eigVal0 = 0 + pdeParam**2
	eigVal1 = 2 + pdeParam**2
	eigVal2 = 6 + pdeParam**2
	for idx in range(lenPtSetOne):
		for jdx in range(lenPtSetTwo):
			kernelMtrx[idx,jdx] = diffKernelFct(ptSetOne[idx,:], ptSetTwo[jdx,:])
		kernelMtrx[idx, lenPtSetTwo] = eigVal0 * sph00(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 1] = eigVal1 * sph10(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 2] = eigVal1 * sph11(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 3] = eigVal1 * sph12(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 4] = eigVal2 * sph20(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 5] = eigVal2 * sph21(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 6] = eigVal2 * sph22(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 7] = eigVal2 * sph23(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
		kernelMtrx[idx, lenPtSetTwo + 8] = eigVal2 * sph24(ptSetOne[idx,0], ptSetOne[idx,1], ptSetOne[idx,2])
	for idx in range(lenPtSetTwo):
		kernelMtrx[lenPtSetOne, idx] = sph00(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 1, idx] = sph10(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 2, idx] = sph11(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 3, idx] = sph12(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 4, idx] = sph20(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 5, idx] = sph21(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 6, idx] = sph22(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 7, idx] = sph23(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
		kernelMtrx[lenPtSetOne + 8, idx] = sph24(ptSetTwo[idx,0], ptSetTwo[idx,1], ptSetTwo[idx,2])
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

