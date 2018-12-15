# NAME: 'kernelFcts.py'
#
# PURPOSE: Collection of different kernel functions
#
# DESCRIPTION: see PURPOSE
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de


from __future__ import division
import numpy as np
import scipy.special

def distSphere(ptOne, ptTwo):
	return 1 - ptOne.dot(ptTwo)







def gaussKernel(ptOne, ptTwo, lengthScale = 1.0):
	distPts = np.linalg.norm(ptOne - ptTwo)
	return np.exp(-distPts**2/(2.0*lengthScale**2))

def maternKernel(ptOne, ptTwo, maternReg, maternScale = 1.0, maternCorrLength = 1.0):
	distPts = np.linalg.norm(ptOne - ptTwo)
	if distPts <= 0:
		return maternScale**2
	else:
		scaledNormOfPts = np.sqrt(2.0*maternReg)*distPts / maternCorrLength
		return maternScale**2 * 2**(1.0-maternReg) / scipy.special.gamma(maternReg) \
			* scaledNormOfPts**(maternReg) * scipy.special.kv(maternReg, scaledNormOfPts)

def expKernel(ptOne, ptTwo, lengthScale = 1.0):
	distPts = np.linalg.norm(ptOne - ptTwo)
	return np.exp(-distPts/(lengthScale))

def tpsKernel(ptOne, ptTwo):
	distPts = np.linalg.norm(ptOne - ptTwo)
	if distPts <= 0:
		return 0
	else:
		return distPts**2 * np.log(distPts)

def tpsKernelSphere(ptOne, ptTwo):
	distPts = distSphere(ptOne, ptTwo)
	if distPts <= 0:
		return 0
	else:
		return distPts**2 * np.log(distPts)

# ptOne and ptTwo are 2-dimensional vectors!
def maternKernelDivFree2d(ptOne, ptTwo, maternReg, maternScale = 1.0, maternCorrLength = 1.0):
	def mAternKernel(pTOne, pTTwo, mAternReg = maternReg, mAternScale = maternScale, mAternCorrLength = maternCorrLength):
		distPts = np.linalg.norm(pTOne - pTTwo)
		if distPts <= 0:
			return maternScale**2
		else:
			scaledNormOfPts = np.sqrt(2.0*mAternReg)*distPts / mAternCorrLength
			return mAternScale**2 * 2**(1.0-mAternReg) / scipy.special.gamma(mAternReg) \
				* scaledNormOfPts**(mAternReg) * scipy.special.kv(mAternReg, scaledNormOfPts)
	kernelIm = np.zeros((2,2))
	diffPts = ptOne - ptTwo
	distPts = np.linalg.norm(diffPts)
	if distPts <= 0:
		kernelIm[0,0] = 0.0
		kernelIm[1,0] = 0.0
		kernelIm[0,1] = 0.0
		kernelIm[1,1] = 0.0
	else:
		kernelIm[0,0] = diffPts[0]**2 * mAternKernel(ptOne, ptTwo, maternReg-2) - mAternKernel(ptOne, ptTwo, maternReg-1)
		kernelIm[1,0] = diffPts[0]*diffPts[1] * mAternKernel(ptOne, ptTwo, maternReg-2) 
		kernelIm[0,1] = diffPts[0]*diffPts[1] * mAternKernel(ptOne, ptTwo, maternReg-2) 
		kernelIm[1,1] = diffPts[1]**2 * mAternKernel(ptOne, ptTwo, maternReg-2) - mAternKernel(ptOne, ptTwo, maternReg-1)
	return kernelIm





