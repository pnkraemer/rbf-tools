# NAME: 'rileyAlgTps.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../modules/')
from ptSetFcts import getPtsHalton
from kernelMtrcs import buildKernelMtrx, buildKernelMtrxCond
from kernelFcts import imqKernel, tpsKernel

import scipy.special
np.set_printoptions(precision = 1)


print("\nHow many points shall we work with? (e.g. 100)")
print("\tnumPts = ?")
numPts = input("Enter: ")
numPts = int(numPts)

print("\nWhich spatial dimension? (e.g. 2)")
print("\tdim = ?")
dim = input("Enter: ")
dim = int(dim)

print("\nWhich shift for Riley? (e.g. 0.001)")
print("\trileyShift = ?")
rileyShift = input("Enter: ")
rileyShift = float(rileyShift)

print("\nWhich accuracy for Riley? (e.g. 1e-08)")
print("\trileyAcc = ?")
rileyAcc = input("Enter: ")
rileyAcc = float(rileyAcc)

print("\nWhich maximal number of iterations? (e.g. 1000)")
print("\trileyNumMaxIt = ?")
rileyNumMaxIt = input("Enter: ")
rileyNumMaxIt = int(rileyNumMaxIt)
print("")


ptSet = getPtsHalton(numPts,dim)

kernelMtrx = buildKernelMtrxCond(ptSet,ptSet, tpsKernel)
kernelMtrxShift = buildKernelMtrxCond(ptSet,ptSet, tpsKernel, rileyShift)

rhs = np.zeros(len(ptSet) + dim + 1)
rhs[0] = 1

trueSol = np.linalg.solve(kernelMtrx,rhs)

startVec = np.linalg.solve(kernelMtrxShift,rhs)
currIt = np.zeros(numPts + dim + 1)
counter = 0
currentRelError = 1.0
relError = np.array(currentRelError)
while currentRelError >= rileyAcc and counter <= rileyNumMaxIt:
	counter = counter + 1
	currIt = startVec + rileyShift * np.linalg.solve(kernelMtrxShift, currIt)
	currentRelError = np.linalg.norm(currIt - trueSol)/np.linalg.norm(trueSol)
	relError = np.append(relError, np.array(currentRelError))

print("Minimal Eigenvalue: ")
print(min(abs(np.linalg.eigvals(kernelMtrx))))


plt.rcParams.update({'font.size': 16})
plt.style.use("ggplot")
plt.figure()
plt.semilogy(relError, '^', label = "N = %i, shift = %.1e"%(numPts, rileyShift))
plt.semilogy(rileyAcc*np.ones(len(relError)),  linewidth = 2, label = "acc = %.1e"%(rileyAcc))
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Relative error")
plt.title("Riley's algorithm (standard)")
plt.legend()
plt.savefig("figures/rileyAlgTps/convergence%i_%.1e.png"%(numPts, rileyShift))
plt.show()


