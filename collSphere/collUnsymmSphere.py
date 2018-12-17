# NAME: 'collUnsymmSphere.py'

# PURPOSE: Check convergence of basic unsymmetric collocation on the sphere

# DESCRIPTION: We build the collocation matrix with higher 
# order thin-plate splines and compare the arising solution 
# to the true solution; differentials are computed with sympy

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de



# 1
# x, y, z, 
# x^2-y^2, 2z^2-x^2-y^2, xy, xz, yz
# x^3, y^3, z^3, x^2y, x^2z, y^2x, y^2z, z^2x, z^2y

from __future__ import division
import numpy as np
import scipy.io as sp
import sympy
from sympy import refine, Q, sqrt
import matplotlib.pyplot as plt
from functools import partial
import sys
sys.path.insert(0,'../')
from kernelFcts import distSphere, tpsKernelSphere
from ptSetFcts import getPtsFibonacciSphere
from kernelMtrcs import buildCollMtrxUnsymmCond, buildKernelMtrxCondSph2



np.set_printoptions(precision=1)


def lapBelOp(exprFct, pdeParam):
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


def lapBelTps(pt1, pt2):
	distPts = distSphere(pt1, pt2, pDeParam)
	return -2*distPts * (distPts + 2*(distPts-1)*np.log(1 - distPts) - 1) -\
		(distPts**2-1)*(2*np.log(1-distPts)+3) +\
		pDeParam*(1-distPts)*(distPts-1)*np.log(1-distPts)


exprTrueSol = sympy.exp(sympy.Symbol('x')) + (1 - sympy.Symbol('x'))
trueSol = sympy.lambdify((pt1, pt2, pt3), exprTrueSol, modules=['numpy']) 
pdeParam = 1.0

lapBelTpsParam = partial(diffTps, pDeParam = pdeParam)

numPts = 50
ptSet = getPtsFibonacciSphere(numPts)
collMtrx = buildCollMtrxUnsymmLapBelCondOrd2(ptSet, ptSet, lapBelTpsParam, pdeParam)


exprRhs = lapBelOp(exprTrueSol, pdeParam)
rhsFct = sympy.lambdify((pt1, pt2, pt3), exprRhs, modules =['numpy'])
rhs = np.zeros(numPts + 9)
for idx in range(numPts):
	rhs[idx] = rhsFct(ptSet[idx,0], ptSet[idx,1], ptSet[idx,2])

lagCoeff = np.linalg.solve(collMtrx, rhs)

numEvalPts = 50
evalPtSet = getPtsFibonacciSphere(numEvalPts, 1)

kernelMtrxLeft = buildKernelMtrxCondSph2(evalPtSet, ptSet, tpsKernelSphere)

approxSol = kernelMtrxLeft.dot(lagCoeff)
approxSol = approxSol[0:numEvalPts]

vecTrueSol = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	vecTrueSol[idx] =  trueSol(evalPtSet[idx,0], evalPtSet[idx,1], evalPtSet[idx,2])

errL2 = np.linalg.norm(vecTrueSol - approxSol) / np.sqrt(numEvalPts)
errLinfty = np.amax(np.fabs(vecTrueSol - approxSol))
print '\nApproximation error:'
print '\tL2 (normalised with sqrt(N))', errL2
print '\tL-infinity =', errLinfty
print ""



























# h1 = get_filldistance(collpoints)
# h2 = get_filldistance(evalpoints)
# print '\nApproximate fill distance of:'
# print '    collpoints: h =', h1
# print '    evalpoints: h =', h2




