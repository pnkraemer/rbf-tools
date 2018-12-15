# NAME: 'lagDecMatern.py'
#
# PURPOSE: Check decay of Lagrange functions
#
# DESCRIPTION: We compute the Lagrange functions for the 
# Matern kernel and plot a) the Lagrange coefficients 
# and b) the Lagrange function
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np 
from kernelFcts import maternKernel
from ptSets import getPtsHalton
from kernelMtrcs import buildKernelMtrx
from functools import partial
import matplotlib.pyplot as plt



np.random.seed(15051994)
np.set_printoptions(precision = 2)
plt.rcParams.update({'font.size': 16})



maternReg = 2.0
maternKernelFixReg = partial(maternKernel, maternReg = maternReg)

dim = 2
numPts = 250
ptSet = getPtsHalton(numPts, dim)

kernelMtrx = buildKernelMtrx(ptSet, ptSet, maternKernelFixReg)
invKernelMtrx = np.linalg.inv(kernelMtrx)
rhs = np.zeros((numPts,1))
rhs[17,0] = 1

# Check decay of Lagrange coefficients
distFrom17PtSet = np.zeros(numPts)
lagCoeff = invKernelMtrx.dot(rhs)
for idx in range(numPts):
	distFrom17PtSet[idx] = np.linalg.norm(ptSet[idx,:] - ptSet[17,:])
	lagCoeff[idx] = np.linalg.norm(lagCoeff[idx])
distSortPtSet = np.argsort(distFrom17PtSet)

# Check decay of Lagrange function
numEvalPts = 250
evalPtSet = np.random.rand(numEvalPts, dim)
evalMtrxLeft = buildKernelMtrx(evalPtSet, ptSet, maternKernelFixReg)
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
plt.savefig("figures/decLagCoeff_250_2.png")
plt.show()



plt.figure()
plt.semilogy(distFrom17EvalPtSet[distSortEvalPtSet], lagFctValues[distSortEvalPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Distance to 17th point")
plt.ylabel("Absolute value of function value")
plt.title("Lagrange function decay")
plt.savefig("figures/decLagFct_250_2.png")
plt.show()








