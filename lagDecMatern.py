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

maternReg = 1.0
maternKernelFixReg = partial(maternKernel, maternReg = maternReg)

dim = 2
numPts = 150
ptSet = getPtsHalton(numPts, dim)


kernelMtrx = buildKernelMtrx(ptSet, ptSet, maternKernelFixReg)
invKernelMtrx = np.linalg.inv(kernelMtrx)

# Check decay of Lagrange coefficients
distVecFrom17 = np.zeros(numPts)
lagCoeff = invKernelMtrx[:,17]
for idx in range(numPts):
	distVecFrom17[idx] = np.linalg.norm(ptSet[idx,:] - ptSet[17,:])
	lagCoeff[idx] = np.linalg.norm(lagCoeff[idx])
distSort = np.argsort(distVecFrom17)

plt.figure()
plt.semilogy(distVecFrom17[distSort], lagCoeff[distSort], 'o', markersize = 8, color = "darkslategray")
plt.grid()
plt.xlabel("Distance from 17th point")
plt.ylabel("Absolute value of coefficient")
plt.title("Lagrange coefficient decay")
plt.show()



numEvalPts = 500
evalPtSet = np.random.rand(numEvalPts, dim)
evalMtrxLeft = buildKernelMtrx(evalPtSet, ptSet, maternKernelFixReg)
lagFctValues = evalMtrxLeft.dot(invKernelMtrx[:,17])
print lagFctValues
distEvalPtsFrom17 = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	distEvalPtsFrom17[idx] = np.linalg.norm(evalPtSet[idx,:] - ptSet[17,:])
	lagFctValues[idx] = np.linalg.norm(lagFctValues[idx])
distSortEval = np.argsort(distEvalPtsFrom17)


plt.figure()
plt.semilogy(distEvalPtsFrom17[distSortEval], lagFctValues[distSortEval], 'o', markersize = 8, color = "darkslategray")
plt.grid()
plt.xlabel("Distance from 17th point")
plt.ylabel("Absolute value of coefficient")
plt.title("Lagrange coefficient decay")
plt.show()








