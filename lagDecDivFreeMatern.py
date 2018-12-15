# NAME: 'lagDecMatern.py'
#
# PURPOSE: Check decay of divergence-free Lagrange functions
#
# DESCRIPTION: We compute the divergence-free Lagrange functions for the 
# Matern kernel and plot a) the Frobenius norm of the Lagrange coefficients 
# and b) the Frobenius norm of the Lagrange function values
# A source: "DIVERGENCE-FREE KERNEL METHODS FOR APPROXIMATING THE STOKES PROBLEM", H. Wendland
# URL: https://epubs.siam.org/doi/pdf/10.1137/080730299
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np 
from kernelFcts import maternKernelDivFree2d
from ptSets import getPtsHalton
from kernelMtrcs import buildKernelMtrxMV
from functools import partial
import matplotlib.pyplot as plt



np.random.seed(15051994)
np.set_printoptions(precision = 2)
plt.rcParams.update({'font.size': 16})

print "\nHow many points shall we compute on? (>25, e.g. 250)"
numPts = input("Enter: ")

print "\nWhich regularity of the Matern function? (>3, e.g. 4.0)"
maternReg = input("Enter: ")

print "\nWhich spatial dimension? (e.g. 2)"
dim = input("Enter: ")

print ""

maternKernelFixRegDF = partial(maternKernelDivFree2d, maternReg = maternReg)

ptSet = getPtsHalton(numPts, dim)

kernelMtrx = buildKernelMtrxMV(ptSet, ptSet, maternKernelFixRegDF)
invKernelMtrx = np.linalg.inv(kernelMtrx)
rhs = np.zeros((2*numPts,2))
rhs[34:36,:] = np.eye(2)

# Check decay of Lagrange coefficients
distFrom17PtSet = np.zeros(numPts)
lagCoeff = invKernelMtrx.dot(rhs)
lagCoeffCompressed = np.zeros(numPts)
for idx in range(numPts):
	distFrom17PtSet[idx] = np.linalg.norm(ptSet[idx,:] - ptSet[17,:])
	lagCoeffCompressed[idx] = np.linalg.norm(lagCoeff[(2*idx):(2*idx + 2),:])
distSortPtSet = np.argsort(distFrom17PtSet)

# Check decay of Lagrange function
numEvalPts = 250
evalPtSet = np.random.rand(numEvalPts, dim)
evalMtrxLeft = buildKernelMtrxMV(evalPtSet, ptSet, maternKernelFixRegDF)
lagFctValues = evalMtrxLeft.dot(invKernelMtrx.dot(rhs))
lagFctValuesCompressed = np.zeros(numEvalPts)
distFrom17EvalPtSet = np.zeros(numEvalPts)
for idx in range(numEvalPts):
	distFrom17EvalPtSet[idx] = np.linalg.norm(evalPtSet[idx,:] - ptSet[17,:])
	lagFctValuesCompressed[idx] = np.linalg.norm(lagFctValues[(2*idx):(2*idx + 2),:])
distSortEvalPtSet = np.argsort(distFrom17EvalPtSet)

plt.figure()
plt.semilogy(distFrom17PtSet[distSortPtSet], lagCoeffCompressed[distSortPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Distance to 17th point")
plt.ylabel("Frobenius norm of coefficient")
plt.legend({"N = %i, nu = %.1f, d = %i"%(numPts, maternReg, dim)}, loc = 4)
plt.title("Divergence-free Lagrange coefficient decay")
plt.savefig("figures/lagDecDivFreeMatern/lagCoeff_%i_%.1f.png"%(numPts, maternReg))
plt.show()



plt.figure()
plt.semilogy(distFrom17EvalPtSet[distSortEvalPtSet], lagFctValuesCompressed[distSortEvalPtSet], 'o', markersize = 8, color = "darkslategray", alpha = 0.8)
plt.grid()
plt.xlabel("Distance to 17th point")
plt.ylabel("Frobenius norm of function value")
plt.legend({"N = %i, nu = %.1f, d = %i"%(numPts, maternReg, dim)}, loc = 4)
plt.title("Divergence-free Lagrange function decay")
plt.savefig("figures/lagDecDivFreeMatern/lagFct_%i_%.1f.png"%(numPts, maternReg))
plt.show()








