# NAME: 'rileyAlgTps.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import sys
sys.path.insert(0,'../modules/')
from ptSetFcts import getPtsHalton
from kernelMtrcs import buildKernelMtrx, buildKernelMtrxCond
from kernelFcts import gaussKernel, tpsKernel, maternKernel, expKernel

import scipy.special
np.set_printoptions(precision = 1)

dim = 2


numReps = 7


def matKernel(pt1, pt2):
	return maternKernel(pt1, pt2, 1.5)

print("\nExponential:")
for i in range(numReps):
	numPts = 2**(i+3)
	ptSet = getPtsHalton(numPts,dim)
	kernelMtrx = buildKernelMtrx(ptSet,ptSet, expKernel, 0.0)
	print("(", numPts, ",", np.linalg.cond(kernelMtrx), ")")

print("\nMatern(nu = 2):")
for i in range(numReps):
	numPts = 2**(i+3)
	ptSet = getPtsHalton(numPts,dim)
	kernelMtrx = buildKernelMtrx(ptSet,ptSet, matKernel, 0.0)
	print("(", numPts, ",", np.linalg.cond(kernelMtrx), ")")

print("\nTPS:")
for i in range(numReps):
	numPts = 2**(i+3)
	ptSet = getPtsHalton(numPts,dim)
	kernelMtrx = buildKernelMtrxCond(ptSet,ptSet, tpsKernel, 0.0)
	print("(", numPts, ",", np.linalg.cond(kernelMtrx), ")")

print("\nGauss:")
for i in range(numReps):
	numPts = 2**(i+3)
	ptSet = getPtsHalton(numPts,dim)
	kernelMtrx = buildKernelMtrx(ptSet,ptSet, gaussKernel, 0.0)
	print("(", numPts, ",", np.linalg.cond(kernelMtrx), ")")












