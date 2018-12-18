# NAME: 'collSymmSphere.py'

# PURPOSE: Check convergence of basic symmetric collocation on the sphere

# DESCRIPTION: We build the collocation matrix with higher 
# order thin-plate splines and compare the arising solution 
# to the true solution; differentials are computed with sympy

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sympy
from functools import partial

import sys
sys.path.insert(0,'../')

from kernelFcts import distSphere, secLapBelTps4Kernel, lapBelTps4Kernel
from ptSetFcts import getPtsFibonacciSphere
from kernelMtrcs import buildSpecialSymCollMtrx, buildSpecialKernelCollMtrx
from miscFcts import computeLapBelOp

np.set_printoptions(precision=1)
np.random.seed(15051994)

print "\nHow many collocation points? (e.g. 150)"
numPts = input("Enter: ")

print "\nWhich PDE parameter? (e.g. 1.0)"
pdeParam = input("Enter: ")
print ""

exprTrueSol = sympy.exp(sympy.Symbol('x')) + (1 - sympy.Symbol('x'))
trueSol = sympy.lambdify((sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')), exprTrueSol, modules=['numpy']) 

secLapBelTps = partial(secLapBelTps4Kernel, pDeParam = pdeParam)
lapBelTps = partial(lapBelTps4Kernel, pDeParam = pdeParam)

ptSet = getPtsFibonacciSphere(numPts)
collMtrx = buildSpecialSymCollMtrx(ptSet, ptSet, secLapBelTps, pdeParam)

exprRhs = computeLapBelOp(exprTrueSol, pdeParam)
rhsFct = sympy.lambdify((sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')), exprRhs, modules =['numpy'])
rhs = np.zeros(numPts + 16)
for idx in range(numPts):
	rhs[idx] = rhsFct(ptSet[idx,0], ptSet[idx,1], ptSet[idx,2])

lagCoeff = np.linalg.solve(collMtrx, rhs)

numEvalPts = 250
evalPtSet = getPtsFibonacciSphere(numEvalPts, 1)
kernelMtrxLeft = buildSpecialKernelCollMtrx(evalPtSet, ptSet, lapBelTps, pdeParam)
approxSol = kernelMtrxLeft.dot(lagCoeff)
approxSol = approxSol[0:numEvalPts]

vecTrueSol = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	vecTrueSol[idx] =  trueSol(evalPtSet[idx,0], evalPtSet[idx,1], evalPtSet[idx,2])

errL2 = np.linalg.norm(vecTrueSol - approxSol) / np.sqrt(numEvalPts)
errLinfty = np.amax(np.fabs(vecTrueSol - approxSol))

print 'Approximation error:'
print '\tl2 (normalised) = %.1e'%errL2
print '\tl-infinity = %.1e'%errLinfty
print ""









