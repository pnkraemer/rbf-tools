# NAME: 'lagDecTps.py'
#
# PURPOSE: Check decay of Lagrange functions with thin-plate splines
#
# DESCRIPTION: We compute the Lagrange functions for the 
# thin-plate spline kernel on the unit sphere S^2 \subset \R^3 
# and plot a) the Lagrange coefficients 
# and b) the Lagrange function
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../')
from kernelFcts import tpsKernelSphere, distSphere
from ptSetFcts import getPtsFibonacciSphere
from kernelMtrcs import buildKernelMtrxCond
from functools import partial


np.random.seed(15051994)
np.set_printoptions(precision = 2)
plt.rcParams.update({'font.size': 16})

print "\nHow many points shall we compute on? (>25, e.g. 250)"
numPts = input("Enter: ")
print ""

dim = 3

ptSet = getPtsFibonacciSphere(numPts)

kernelMtrx = buildKernelMtrxCond(ptSet, ptSet, tpsKernelSphere)
invKernelMtrx = np.linalg.inv(kernelMtrx)
rhs = np.zeros((numPts + dim + 1,1))
rhs[17,0] = 1

# Check decay of Lagrange coefficients
distFrom17PtSet = np.zeros(numPts)
lagCoeff = invKernelMtrx.dot(rhs)
for idx in range(numPts):
	distFrom17PtSet[idx] = distSphere(ptSet[idx,:], ptSet[17,:])
	lagCoeff[idx] = np.linalg.norm(lagCoeff[idx])
distSortPtSet = np.argsort(distFrom17PtSet)

# Check decay of Lagrange function
numEvalPts = 250
evalPtSet = getPtsFibonacciSphere(numEvalPts)
evalMtrxLeft = buildKernelMtrxCond(evalPtSet, ptSet, tpsKernelSphere)
lagFctValues = evalMtrxLeft.dot(invKernelMtrx.dot(rhs))
distFrom17EvalPtSet = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	distFrom17EvalPtSet[idx] = distSphere(evalPtSet[idx,:], ptSet[17,:])
	lagFctValues[idx] = np.linalg.norm(lagFctValues[idx])
distSortEvalPtSet = np.argsort(distFrom17EvalPtSet)

plt.figure()
plt.semilogy(distFrom17PtSet[distSortPtSet], lagCoeff[distSortPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Geodesic distance to 18th point")
plt.ylabel("Absolute value of coefficient")
plt.title("Lagrange coefficient decay")
plt.legend({"N = %i"%(numPts)})
plt.savefig("figures/lagDecTpsSphere/lagCoeff_%i.png"%(numPts))
plt.show()



plt.figure()
plt.semilogy(distFrom17EvalPtSet[distSortEvalPtSet], lagFctValues[distSortEvalPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Geodesic distance to 18th point")
plt.ylabel("Absolute value of function value")
plt.legend({"N = %i"%(numPts)})
plt.title("Lagrange function decay")
plt.savefig("figures/lagDecTpsSphere/lagFct_%i.png"%(numPts))
plt.show()








