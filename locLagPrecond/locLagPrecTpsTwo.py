
# NAME: 'locLagPrecTpsTwo.py'
#
# PURPOSE: Check effect of local Lagrange preconditioner on condition number
#
# DESCRIPTION: We compute the local Lagrange function preconditioner with 
# thin-plate spline interpolation on the sphere and 
# check its influence on the condition number of the kernel matrix.
# We show that GMRES solver roughly takes a constant number of iterations,
# even for increasing pointset sizes
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de


import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.sparse.linalg as spla

import sys
sys.path.insert(0,'../')
from kernelFcts import tpsKernel
from ptSetFcts import getPtsFibonacciSphere
from kernelMtrcs import buildKernelMtrxCond

class gmresCounter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.numIter = 0
    def __call__(self, rk=None):
        self.numIter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def locLag(ptSet, tree, radius, kernelFct):

	numPts = len(ptSet)
	numNeighb = 1.0 * radius * np.log10(numPts) * np.log10(numPts)
	numNeighb = np.minimum(np.floor(numNeighb), numPts)

	preconMtrx = np.zeros((numPts + 4, numPts))
	for idx in range(numPts):

		distNeighb, indNeighb = tree.query(ptSet[idx], k = numNeighb)

		locKernelMtrx = buildKernelMtrxCond(ptSet[indNeighb], ptSet[indNeighb], kernelFct)
		locRhs = np.zeros(len(indNeighb) + 4)
		locRhs[(indNeighb==idx*np.ones((1, len(indNeighb)))[0]).nonzero()] = 1

		locCoeff = np.linalg.solve(locKernelMtrx, locRhs)

		localCoeff = locCoeff[range(len(indNeighb))]
		locPolyBlock = locCoeff[range(len(indNeighb), len(indNeighb) + 4)]

		preconMtrx[indNeighb, idx] = localCoeff.T
		preconMtrx[numPts:(numPts+4), idx] = locPolyBlock.T

	return preconMtrx

def rhsFct(x,y,z):
	return np.exp(x)*(1-x)





print "\nHow many interpolation points? (e.g. 150)"
numPts = input("Enter: ")
print "\nWhich local radius? (e.g. 7.0)"
locRadius = input("Enter: ")
print "\nWhich GMRES tolerance? (e.g. 1e-10)"
gmresTol = input("Enter: ")
print ""

ptSet = getPtsFibonacciSphere(numPts)
kdTree = scipy.spatial.KDTree(ptSet)


kernelMtrx = buildKernelMtrxCond(ptSet, ptSet, tpsKernel)
preconMtrx = locLag(ptSet, kdTree, locRadius, tpsKernel)
conditionedMtrx = kernelMtrx.dot(preconMtrx)
conditionedMtrx = conditionedMtrx[0:numPts, 0:numPts]

print 'Effect of preconditioner:'
print '\tcond(K) =', np.linalg.cond(kernelMtrx)
print '\tcond(KP) =', np.linalg.cond(conditionedMtrx)

rhsVec = np.zeros(numPts)
for idx in range(numPts):
	rhsVec[idx] = rhsFct(ptSet[idx,0], ptSet[idx,1], ptSet[idx,2])

gmresCtr = gmresCounter()
solVec, info = spla.gmres(conditionedMtrx, rhsVec, tol = gmresTol, callback = gmresCtr)
print "\nNumber of iterations for GMRES: "
print '\tnumIt =', gmresCtr.numIter
print ""


