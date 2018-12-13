# NAME: 'interpolMatern1d.py'

# PURPOSE: Basic 1-dimensional interpolation using Matern functions

# DESCRIPTION: I solve a system involving a Matern-kernel matrix 
# where the Matern kernel is based on scipy.special's functions (Bessel-fct. and Gamma-fct.)
# and plot the 10-th Lagrange function (10th for visibility reasons in the plot).

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy
import scipy.special
import matplotlib.pyplot

numpy.set_printoptions(precision = 1)

print "\nHow many points shall we work with? (e.g. 120)"
print "\tnumPts = ?"
numPts = input("Enter: ")

print "\nWhich regularity of the Matern function? (e.g. 2.0)"
print "\tmaternReg = ?"
maternReg = input("Enter: ")

print ""
print "(Please change other parameters and choices of pointsets in the code)"
print ""

maternScale = 1.0
maternCorrLength = 1.0
maternReg = 1.0

def maternKernFct(firstPt, secondPt, mAternReg = maternReg, mAternScale = maternScale, mAternCorrLength = maternCorrLength):
	normOfPts = numpy.linalg.norm(firstPt - secondPt)
	if normOfPts <= 0:
		return maternScale**2
	else:
		scaledNormOfPts = numpy.sqrt(2.0*mAternReg)*normOfPts / mAternCorrLength
		return mAternScale**2 * 2**(1.0-mAternReg) / scipy.special.gamma(mAternReg) * scaledNormOfPts**(mAternReg) * scipy.special.kv(mAternReg, scaledNormOfPts)

def buildKernelMtrx(firstSet, secondSet, kernFct = maternKernFct):
	mtrxFrame = numpy.zeros((len(firstSet),len(secondSet)))
	for idx in range(len(firstSet)):
		for jdx in range(len(secondSet)):
			mtrxFrame[idx,jdx] = kernFct(firstSet[idx], secondSet[jdx])
	return mtrxFrame


#ptSet = numpy.random.rand(numPts, 1)
ptSet = numpy.linspace(0,1,numPts)

kernMtrx = buildKernelMtrx(ptSet,ptSet)

rhs = numpy.zeros(len(ptSet))
rhs[10] = 1

trueSol = numpy.linalg.solve(kernMtrx,rhs)

evalPtSet = numpy.linspace(0,1,500)
evalMtrxLeft = buildKernelMtrx(evalPtSet, ptSet)
LagFunc = evalMtrxLeft.dot(trueSol)
matplotlib.pyplot.plot(evalPtSet, LagFunc, linewidth = 4, color = "darkslategray")
matplotlib.pyplot.plot(ptSet, rhs, 'o', color = "red")
matplotlib.pyplot.title("'10th' Lagrange function")
matplotlib.pyplot.xlim((-0.1,1.1))
matplotlib.pyplot.ylim((-0.1,1.1))
matplotlib.pyplot.grid()
matplotlib.pyplot.show()




