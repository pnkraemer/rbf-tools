# NAME: 'rileyAlgMatern.py'

# PURPOSE: Check the influence of the shifts 
# and accuracy onto Riley's Algorithm


# DESCRIPTION: I solve a system involving a Matern-kernel matrix 
# and iteratively compute the approximations with Riley's algorithms
# (as in (2) on slide 42 on https://drna.padovauniversitypress.it/system/files/papers/Fasshauer-2008-Lecture3.pdf)

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import sympy
import numpy
import matplotlib.pyplot
from halton import halton_sequence
import scipy.special

numpy.set_printoptions(precision = 1)

print "\nHow many points shall we work with? (e.g. 100)"
print "\tnumPts = ?"
numPts = input("Enter: ")

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


maternScale = 1.0
maternCorrLength = 1.0
maternReg = 1.0

def maternkernel(firstPt, secondPt, mAternReg = maternReg, mAternScale = maternScale, mAternCorrLength = maternCorrLength):
	normOfPts = numpy.sqrt(numpy.abs(firstPt)**(2.0) + numpy.abs(secondPt)**(2.0))
	if normOfPts <= 0:
		return maternScale**2
	else:
		scaledNormOfPts = numpy.sqrt(2.0*mAternReg)*normOfPts / mAternCorrLength
		return mAternScale**2 * 2**(1.0-mAternReg) / scipy.special.gamma(mAternReg) * scaledNormOfPts**(mAternReg) * scipy.special.kv(mAternReg, scaledNormOfPts)

def buildKernel(X,Y, kernelfct = maternkernel):
	XX = numpy.zeros((len(X),len(Y)))
	for i in range(len(X)):
		for j in range(len(Y)):
			dummy = X[i] - Y[j]
			XX[i,j] = kernelfct(dummy[0], dummy[1])
	return XX


def shiftKernelMtrx(kernelMtrx, rileyShift = rileyShift):
	identity = numpy.identity(len(kernelMtrx))
	return kernelMtrx + rileyShift * identity


ptSet = halton_sequence(numPts + 1,2)
ptSet = ptSet[1:,:]

kernMtrx = buildKernel(ptSet,ptSet)
shiftKernMtrx = shiftKernelMtrx(kernMtrx)

rhs = numpy.zeros(numPts)
rhs[0] = 1

trueSol = numpy.linalg.solve(kernMtrx,rhs)

print "\nIterations - relative errors:"
startVec = numpy.linalg.solve(shiftKernMtrx,rhs)
currIt = numpy.zeros(numPts)
counter = 0
currentRelError = 100.0
while currentRelError >= rileyAcc and counter <= rileyNumMaxIt:
	counter = counter + 1
	currIt = startVec + rileyShift * numpy.linalg.solve(shiftKernMtrx,currIt)
	currentRelError = numpy.linalg.norm(currIt - trueSol)/numpy.linalg.norm(trueSol)
	print counter, "-", '{:.5e}'.format(currentRelError)
print ""


