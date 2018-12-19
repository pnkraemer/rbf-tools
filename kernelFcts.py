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


def imqKernel(ptOne, ptTwo, imqPower = 1.0):
	distPts = np.linalg.norm(ptOne - ptTwo)
	return 1.0 / np.sqrt((1 + distPts**2)**imqPower)

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

def tps3KernelSphere(ptOne, ptTwo):
	distPts = distSphere(ptOne, ptTwo)
	if distPts <= 0:
		return 0
	else:
		return distPts**3 * np.log(distPts)

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

# Lap.-Bel. Operator of (1-s)**3 * log(1-s)
def lapBelTps3Kernel(pt1, pt2, pDeParam = 1.0):
	distPts = distSphere(pt1, pt2)
	if distPts <= 0:
		return 0
	else:
		lapBel = 2*(1 - distPts)*(- distPts)**2*(3*np.log(distPts) + 1) + ((1 - distPts)**2 - 1)*(5*(1 - distPts) + 6*(- distPts)*np.log(distPts) - 5)
		return -lapBel + pDeParam**2 * (distPts)**3 * np.log(distPts)

# Lap.-Bel. Operator of (1-s)**4 * log(1-s)
def lapBelTps4Kernel(pt1, pt2, pDeParam = 1.0):
	distPts = distSphere(pt1, pt2)
	if distPts <= 0:
		return 0
	else:
		lapBel = -(- distPts)**2*(2*(1 - distPts)*(- distPts)*(4*np.log(distPts) + 1) + ((1 - distPts)**2 - 1)*(12*np.log(distPts) + 7))
		return -lapBel +pDeParam**2 * (distPts)**4 * np.log(distPts)

# Lap.-Bel. Operator of (Lap.-Bel. Operator of (1-s)**4 * log(1-s))
def secLapBelTps4Kernel(pt1, pt2, pDeParam = 1.0):
	distPts = distSphere(pt1, pt2)
	if distPts <= 0:
		return 0
	else:
		lapBel = -(- distPts)**2*(2*(1 - distPts)*(- distPts)*(4*np.log(distPts) + 1) + ((1 - distPts)**2 - 1)*(12*np.log(distPts) + 7))
		secLapBel = 400*(1 - distPts)**4*np.log(distPts) + 360*(1 - distPts)**4 - 576*(1 - distPts)**3*np.log(distPts) - 416*(1 - distPts)**3 - 96*(1 - distPts)**2*np.log(distPts) - 240*(1 - distPts)**2 + 320*(1 - distPts)*np.log(distPts) + 288*(1 - distPts) - 48*np.log(distPts) + 8
		return secLapBel -2* pDeParam**2 * lapBel + pDeParam**4 * (distPts)**4 * np.log(distPts)






