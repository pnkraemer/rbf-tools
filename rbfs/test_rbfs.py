############################################
# NAME:
# test_rbfs.py
#
# PURPOSE:
# Test the functions in rbfs.py
#
# DESCRIPTION:
# ??
#
# AUTHOR: 
# Nicholas Kraemer, kraemer@ins.uni-bonn.de
#
############################################


from __future__ import division	#division of integers into decimal
import numpy as np 
import rbfs


np.set_printoptions(precision=2)


print "\nHow many total points?"
numPts = input("\tEnter: ")

print "\nWhich dimension?"
dim = input("\tEnter: ")

ptset = np.random.rand(numPts, dim)

# Choose some hyperparameters for the Matern kernel
smoothPar = 2.0
lengthScalePar = 1.0
scalingPar = 1.0




# turn rbf and distance into kernelfunction
def maternkernel(firstpt, secondpt, whichNorm = np.inf, sMoothPar = smoothPar, lgthSclPar = lengthScalePar, sCalingPar = scalingPar):
	distance = rbfs.normDist(firstpt, secondpt, whichNorm)
	return rbfs.maternfct(distance, sMoothPar, lgthSclPar, sCalingPar)


# build kernelmatrix
kernMtrx = rbfs.buildKernMtrxStd(ptset, ptset, maternkernel)

print "\nKern matrix: "

print "\tK =", kernMtrx
print ""



