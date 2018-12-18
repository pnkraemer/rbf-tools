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
		s = 1 - distPts
		lapBel = 2*s*(s - 1)**2*(3*np.log(-s + 1) + 1) + (s**2 - 1)*(5*s + 6*(s - 1)*np.log(-s + 1) - 5)
		return -lapBel + pDeParam**2 * (1 - s)**3 * np.log(1-s)

# Lap.-Bel. Operator of (1-s)**4 * log(1-s)
def lapBelTps4Kernel(pt1, pt2, pDeParam = 1.0):
	distPts = distSphere(pt1, pt2)
	if distPts <= 0:
		return 0
	else:
		s = 1 - distPts
		lapBel = -(s - 1)**2*(2*s*(s - 1)*(4*np.log(-s + 1) + 1) + (s**2 - 1)*(12*np.log(-s + 1) + 7))
		return -lapBel +pDeParam**2 * (1 - s)**4 * np.log(1-s)

# Lap.-Bel. Operator of (Lap.-Bel. Operator of (1-s)**4 * log(1-s))
def secLapBelTps4Kernel(pt1, pt2, pDeParam = 1.0):
	distPts = distSphere(pt1, pt2)
	if distPts <= 0:
		return 0
	else:
		s = 1 - distPts
		lapBel = -(s - 1)**2*(2*s*(s - 1)*(4*np.log(-s + 1) + 1) + (s**2 - 1)*(12*np.log(-s + 1) + 7))
		secLapBel = 400*s**4*np.log(-s + 1) + 360*s**4 - 576*s**3*np.log(-s + 1) - 416*s**3 - 96*s**2*np.log(-s + 1) - 240*s**2 + 320*s*np.log(-s + 1) + 288*s - 48*np.log(-s + 1) + 8
		return secLapBel -2* pDeParam**2 * lapBel + pDeParam**4 * (1 - s)**4 * np.log(1-s)






