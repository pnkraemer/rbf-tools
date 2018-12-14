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

def gaussKernel(ptOne, ptTwo, lengthScale = 1.0):
	distPts = np.linalg.norm(ptOne - ptTwo)
	return np.exp(-distPts**2/(2.0*lengthScale**2))

def maternKernel(firstPt, secondPt, maternReg, maternScale = 1.0, maternCorrLength = 1.0):
	distPts = np.linalg.norm(firstPt - secondPt)
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

