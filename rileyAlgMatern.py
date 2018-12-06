# NAME:

# PURPOSE:

# DESCRIPTION:

# AUTHOR:

import sympy
import numpy
import matplotlib.pyplot
from halton import halton_sequence
import scipy.special


numpy.set_printoptions(precision = 1)

# Determine size of the checkup
print "\nHow many points shall we work with? (e.g. 100)"
print "\tnumPts = ?"
numPts = input("Enter: ")

print "\nWhich regularity for the Matern function? (e.g. 1)"
print "\tmaternReg = ?"
maternReg = input("Enter: ")

print "\nWhich shift for Riley? (e.g. 0.01)"
print "\trileyShift = ?"
rileyShift = input("Enter: ")

print "\nWhich accuracy for Riley? (e.g. 1e-08)"
print "\trileyAcc = ?"
rileyAcc = input("Enter: ")

print "\nWhich maximal number of iterations? (e.g. 1000)"
print "\trileyNumMaxIt = ?"
rileyNumMaxIt = input("Enter: ")
print ""


sigma = 1.0
rho = 1.0

# computes l2 norm of (xxx,yyy) and returns matern function
def maternkernel(xx, yy, nU = maternReg, sIgma = sigma, rHo = rho):
	xxx = numpy.abs(xx)
	yyy = numpy.abs(yy)
	r = numpy.sqrt(xxx**(2.0) + yyy**(2.0))
	if r <= 0:
		return sIgma**2
	else:
		z = numpy.sqrt(2.0*nU)*r / rHo
		a = sIgma**2 * 2**(1.0-nU) / scipy.special.gamma(nU)
		return a * z**(nU) * scipy.special.kv(nU, z)



# Function to build a matrix for divergence-free kernels
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



# Build pointset (Halton)
ptSet = halton_sequence(numPts + 1,2)
ptSet = ptSet[1:,:]

# Build divergence-free kernel matrix
kernMtrx = buildKernel(ptSet,ptSet)
shiftKernMtrx = shiftKernelMtrx(kernMtrx)

# Construct Lagrange functions, use random evaluation points
invKernMtrx = numpy.linalg.inv(kernMtrx)
invShiftKernMtrx = numpy.linalg.inv(shiftKernMtrx)


rhs = numpy.zeros(numPts)
rhs[0] = 1




trueSol = invKernMtrx.dot(rhs)

print "\nIterations - relative errors:"

startVec = invShiftKernMtrx.dot(rhs)
currIt = startVec
counter = 0
currentRelError = 100.0


while currentRelError >= rileyAcc and counter <= rileyNumMaxIt:
	counter = counter + 1
	#print iteration
	currIt = startVec + rileyShift * invShiftKernMtrx.dot(currIt)
	currentRelError = numpy.linalg.norm(currIt - trueSol)/numpy.linalg.norm(trueSol)
	print counter, "-", '{:.1e}'.format(currentRelError)


print ""


