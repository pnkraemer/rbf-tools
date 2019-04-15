# NAME: 'rileyAlgTps.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm

# DESCRIPTION: I solve a system involving an exponential kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
# import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../modules/')
from ptSetFcts import getPtsHalton
from kernelMtrcs import buildKernelMtrx, buildKernelMtrxCond
from kernelFcts import maternKernel, tpsKernel

import scipy.special
np.set_printoptions(precision = 1)

def matKernel(pt1, pt2):
	return maternKernel(pt1, pt2, 1.5)
# print("\nHow many points shall we work with? (e.g. 100)")
# print("\tnumPts = ?")
# numPts = input("Enter: ")
# numPts = int(numPts)

# print("\nWhich spatial dimension? (e.g. 2)")
# print("\tdim = ?")
# dim = input("Enter: ")
# dim = int(dim)

# print("\nWhich shift for Riley? (e.g. 0.001)")
# print("\trileyShift = ?")
# rileyShift = input("Enter: ")
# rileyShift = float(rileyShift)

# print("\nWhich accuracy for Riley? (e.g. 1e-08)")
# print("\trileyAcc = ?")
# rileyAcc = input("Enter: ")
# rileyAcc = float(rileyAcc)

# print("\nWhich maximal number of iterations? (e.g. 1000)")
# print("\trileyNumMaxIt = ?")
# rileyNumMaxIt = input("Enter: ")
# rileyNumMaxIt = int(rileyNumMaxIt)
# print("")

numPts = 500;
dim = 2;
rileyShift = 1e-07
rileyNumMaxIt = 1000
rileyAcc = 1e-10

ptSet = getPtsHalton(numPts,dim)

kernelMtrx = buildKernelMtrx(ptSet,ptSet, matKernel)
kernelMtrxShift = buildKernelMtrx(ptSet,ptSet, matKernel, rileyShift)

rhs = np.zeros(len(ptSet))
rhs[0] = 1

trueSol = np.linalg.solve(kernelMtrx,rhs)

startVec = np.linalg.solve(kernelMtrxShift,rhs)
currIt = np.copy(startVec)
counter = 0
currentRelError = np.linalg.norm(kernelMtrx.dot(currIt) - rhs)/np.linalg.norm(rhs)
relError = np.array(currentRelError)
print("Matern: shift = %.1f", rileyShift)
while currentRelError >= rileyAcc and counter <= rileyNumMaxIt:
	print("(", counter, ",", currentRelError, ")")
	counter = counter + 1
	currIt = startVec + rileyShift * np.linalg.solve(kernelMtrxShift, currIt)
	currentRelError = np.linalg.norm(kernelMtrx.dot(currIt) - rhs)/np.linalg.norm(rhs)
	relError = np.append(relError, np.array(currentRelError))
print("(", counter, ",", currentRelError, ")")
