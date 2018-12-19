# NAME: 'miscFcts.py'
#
# PURPOSE: Collect all support functions that do not fit the other categories
#
# DESCRIPTION: see PURPOSE; we collect e.g. sympy-based functions, fill distances, Lebesgue constants
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import sympy
import numpy as np
import scipy.spatial

from kernelFcts import distSphere


def locLagPrecon(ptSet, tree, radius, kernelMtrxFct, kernelFct, polBlockSize):

	numPts = len(ptSet)
	numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
	numNeighb = np.minimum(np.floor(numNeighb), numPts)

	preconMtrx = np.zeros((numPts + polBlockSize, numPts))
	for idx in range(numPts):

		distNeighb, indNeighb = tree.query(ptSet[idx], k = numNeighb)

		locKernelMtrx = kernelMtrxFct(ptSet[indNeighb], ptSet[indNeighb], kernelFct)
		locRhs = np.zeros(len(indNeighb) + polBlockSize)
		locRhs[(indNeighb==idx*np.ones((1, len(indNeighb)))[0]).nonzero()] = 1

		locCoeff = np.linalg.solve(locKernelMtrx, locRhs)

		locCoeffBlock = locCoeff[range(len(indNeighb))]
		locPolyBlock = locCoeff[range(len(indNeighb), len(indNeighb) + polBlockSize)]
		preconMtrx[indNeighb, idx] = locCoeffBlock.T
		preconMtrx[numPts:(numPts + polBlockSize), idx] = locPolyBlock.T
	return preconMtrx, numNeighb





def locLagPreconColl(ptSet, tree, radius, kernelMtrxFct, kernelFct, pdeParam , polBlockSize):

	numPts = len(ptSet)
	numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
	numNeighb = np.minimum(np.floor(numNeighb), numPts)

	preconMtrx = np.zeros((numPts + polBlockSize, numPts))
	for idx in range(numPts):

		distNeighb, indNeighb = tree.query(ptSet[idx], k = numNeighb)

		locKernelMtrx = kernelMtrxFct(ptSet[indNeighb], ptSet[indNeighb], kernelFct, pdeParam)
		locRhs = np.zeros(len(indNeighb) + polBlockSize)
		locRhs[(indNeighb==idx*np.ones((1, len(indNeighb)))[0]).nonzero()] = 1

		locCoeff = np.linalg.solve(locKernelMtrx, locRhs)

		locCoeffBlock = locCoeff[range(len(indNeighb))]
		locPolyBlock = locCoeff[range(len(indNeighb), len(indNeighb) + polBlockSize)]
		preconMtrx[indNeighb, idx] = locCoeffBlock.T
		preconMtrx[numPts:(numPts + polBlockSize), idx] = locPolyBlock.T
	return preconMtrx, numNeighb



def computeLapBelOp(exprFct, pdeParam):
	exprX = sympy.sin(sympy.Symbol('t')) * sympy.cos(sympy.Symbol('p'))
	exprY = sympy.sin(sympy.Symbol('t')) * sympy.sin(sympy.Symbol('p'))
	exprZ = sympy.cos(sympy.Symbol('t'))
	exprFctPol = exprFct.subs({'x': exprX, 'y': exprY, 'z': exprZ})

	lapBelOpFctPol = sympy.diff(sympy.sin(sympy.Symbol('t')) * sympy.diff(exprFctPol,sympy.Symbol('t')), \
		sympy.Symbol('t')) / (sympy.sin(sympy.Symbol('t'))) +\
		sympy.diff(exprFctPol, sympy.Symbol('p'), sympy.Symbol('p')) / (sympy.sin(sympy.Symbol('t'))**2)

	exprT = sympy.acos(sympy.Symbol('z'))
	exprP = sympy.atan2(sympy.Symbol('y'), sympy.Symbol('x'))
	exprLapBelCart = lapBelOpFctPol.subs({'t': exprT, 'p': exprP})

	return -exprLapBelCart + pdeParam**2 * exprFct


def getApproxFillDistanceSphere(ptSet, numSamp = 1000, distFct = distSphere):
	evalPtSet = getPtsFibonacciSphere(numSamp, 1)
	distMtrx = buildKernelMtrx(ptSet, evalPtSet, distFct)
	return np.amax(distMtrx.min(axis = 0))

def sph00(pt1, pt2, pt3):
	return 1.0
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
def sph30(pt1, pt2, pt3):
	return (3*pt1**2 - pt2**2)*pt2
def sph31(pt1, pt2, pt3):
	return pt1*pt2*pt3
def sph32(pt1, pt2, pt3):
	return pt2*(4*pt3**2 - pt1**2 - pt2**2)
def sph33(pt1, pt2, pt3):
	return pt3*(2*pt3**2 - 3*pt1**2 - 3*pt2**2)
def sph34(pt1, pt2, pt3):
	return pt1*(4*pt3**2 - pt1**2 - pt2**2)
def sph35(pt1, pt2, pt3):
	return pt3*(pt1**2 - pt2**2)
def sph36(pt1, pt2, pt3):
	return pt1*(pt1**2 - 3*pt2**2)

class gmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.numIter = 0
    def __call__(self, rk=None):
        self.numIter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



