# NAME: 'rileyAlgResBased.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto the residual-based version of Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 55 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../modules/')
from ptSetFcts import getPtsHalton
from kernelMtrcs import buildKernelMtrx, buildKernelMtrxShift
from kernelFcts import expKernel

import scipy.special

np.set_printoptions(precision = 1)
plt.rcParams.update({'font.size': 16})

print "\nHow many points shall we work with? (e.g. 100)"
print "\tnumPts = ?"
numPts = input("Enter: ")

print "\nWhich spatial dimension? (e.g. 2)"
print "\tdim = ?"
dim = input("Enter: ")

print "\nWhich shift for Riley? (e.g. 0.001)"
print "\trileyShift = ?"
rileyShift = input("Enter: ")

print "\nWhich accuracy for Riley? (e.g. 1e-08)"
print "\trileyAcc = ?"
rileyAcc = input("Enter: ")

print "\nWhich maximal number of iterations? (e.g. 1000)"
print "\trileyNumMaxIt = ?"
rileyNumMaxIt = input("Enter: ")
print ""


ptSet = getPtsHalton(numPts,dim)


kernelMtrx = buildKernelMtrx(ptSet,ptSet, expKernel)
kernelMtrxShift = buildKernelMtrxShift(ptSet,ptSet, expKernel, rileyShift)

rhs = np.zeros(len(ptSet))
rhs[0] = 1

trueSol = np.linalg.solve(kernelMtrx,rhs)

startVec = np.linalg.solve(kernelMtrxShift,rhs)
currIt = np.zeros(numPts)
counter = 0
currentRelError = 1.0
relError = np.array(currentRelError)
while currentRelError >= rileyAcc and counter <= rileyNumMaxIt:
	counter = counter + 1
	residual = rhs - kernelMtrx.dot(currIt)
	currIt = currIt + np.linalg.solve(kernelMtrxShift, residual)
	currentRelError = np.linalg.norm(currIt - trueSol)/np.linalg.norm(trueSol)
	relError = np.append(relError, np.array(currentRelError))




plt.figure()
plt.semilogy(relError, '^', markersize = 12, color = "darkslategray", alpha = 0.9, label = "N = %i, shift = %.1e"%(numPts, rileyShift))
plt.semilogy(rileyAcc*np.ones(len(relError)), color = 'red', linewidth = 2, label = "acc = %.1e"%(rileyAcc))
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Relative error")
plt.title("Riley's algorithm (residual-based)")
plt.legend()
plt.savefig("figures/rileyAlgResBased/convergence%i_%.1e.png"%(numPts, rileyShift))
plt.show()









