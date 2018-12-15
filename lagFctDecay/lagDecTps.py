# NAME: 'lagDecTps.py'
#
# PURPOSE: Check decay of Lagrange functions with thin-plate splines
#
# DESCRIPTION: We compute the Lagrange functions for the 
# thin-plate spline kernel and plot a) the Lagrange coefficients 
# and b) the Lagrange function
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../')
from kernelFcts import tpsKernel
from ptSetFcts import getPtsHalton
from kernelMtrcs import buildKernelMtrxCond
from functools import partial


np.random.seed(15051994)
np.set_printoptions(precision = 2)
plt.rcParams.update({'font.size': 16})

print "\nHow many points shall we compute on? (>25, e.g. 250)"
numPts = input("Enter: ")

print "\nWhich spatial dimension? (e.g. 2)"
dim = input("Enter: ")

print ""

ptSet = getPtsHalton(numPts, dim)

kernelMtrx = buildKernelMtrxCond(ptSet, ptSet, tpsKernel)
invKernelMtrx = np.linalg.inv(kernelMtrx)
rhs = np.zeros((numPts + dim + 1,1))
rhs[17,0] = 1

# Check decay of Lagrange coefficients
distFrom17PtSet = np.zeros(numPts)
lagCoeff = invKernelMtrx.dot(rhs)
for idx in range(numPts):
	distFrom17PtSet[idx] = np.linalg.norm(ptSet[idx,:] - ptSet[17,:])
	lagCoeff[idx] = np.linalg.norm(lagCoeff[idx])
distSortPtSet = np.argsort(distFrom17PtSet)

# Check decay of Lagrange function
numEvalPts = numPts + 1
evalPtSet = np.random.rand(numEvalPts, dim)
evalMtrxLeft = buildKernelMtrxCond(evalPtSet, ptSet, tpsKernel)
lagFctValues = evalMtrxLeft.dot(invKernelMtrx.dot(rhs))
distFrom17EvalPtSet = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	distFrom17EvalPtSet[idx] = np.linalg.norm(evalPtSet[idx,:] - ptSet[17,:])
	lagFctValues[idx] = np.linalg.norm(lagFctValues[idx])
distSortEvalPtSet = np.argsort(distFrom17EvalPtSet)

plt.figure()
plt.semilogy(distFrom17PtSet[distSortPtSet], lagCoeff[distSortPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Distance to 17th point")
plt.ylabel("Absolute value of coefficient")
plt.title("Lagrange coefficient decay")
plt.legend({"N = %i, d = %i"%(numPts, dim)})
plt.savefig("figures/lagDecTps/lagCoeff_%i.png"%(numPts))
plt.show()



plt.figure()
plt.semilogy(distFrom17EvalPtSet[distSortEvalPtSet], lagFctValues[distSortEvalPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Distance to 17th point")
plt.ylabel("Absolute value of function value")
plt.legend({"N = %i, d = %i"%(numPts, dim)})
plt.title("Lagrange function decay")
plt.savefig("figures/lagDecTps/lagFct_%i.png"%(numPts))
plt.show()








