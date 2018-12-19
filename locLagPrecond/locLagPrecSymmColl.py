# NAME: 'locLagPrecSymmColl.py'
#
# PURPOSE: Check effect of local Lagrange preconditioner on condition number
# of collocation matrices
#
# DESCRIPTION: We compute the local Lagrange function preconditioner with 
# higher order thin-plate spline collocation (symmetric) on the sphere and 
# check its influence on the condition number of the collocation matrix.
# We show that GMRES solver roughly takes a constant number of iterations,
# even for increasing pointset sizes
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.sparse.linalg as spla
from functools import partial
import sys
sys.path.insert(0,'../modules/')
from kernelFcts import secLapBelTps4Kernel
from ptSetFcts import getPtsFibonacciSphere
from kernelMtrcs import buildSpecialSymCollMtrx
from miscFcts import locLagPreconColl, gmresCounter


def rhsFct(x,y,z):
	return np.exp(x)*(1-x)

print "\nHow many interpolation points? (e.g. 150)"
numPts = input("Enter: ")
print "\nWhich local radius? (e.g. 7.0)"
locRadius = input("Enter: ")
print "\nWhich GMRES tolerance? (e.g. 1e-10)"
gmresTol = input("Enter: ")
print "\nWhich PDE parameter? (e.g. 1.0)"
pdeParam = input("Enter: ")
print ""

secLapBelTps = partial(secLapBelTps4Kernel, pDeParam = pdeParam)

ptSet = getPtsFibonacciSphere(numPts)
kdTree = scipy.spatial.KDTree(ptSet)

collMtrx = buildSpecialSymCollMtrx(ptSet, ptSet, secLapBelTps, pdeParam)

preconMtrx, numNeighb = locLagPreconColl(ptSet, kdTree, locRadius, buildSpecialSymCollMtrx, secLapBelTps, pdeParam, 16)
conditionedMtrx = collMtrx.dot(preconMtrx)
conditionedMtrx = conditionedMtrx[0:numPts, 0:numPts]

print 'Number of neighbors for localisation:'
print '\tn =', numNeighb

print '\nEffect of preconditioner:'
print '\tcond(K) =', np.linalg.cond(collMtrx)
print '\tcond(KP) =', np.linalg.cond(conditionedMtrx)

rhsVec = np.zeros(numPts)
for idx in range(numPts):
	rhsVec[idx] = rhsFct(ptSet[idx,0], ptSet[idx,1], ptSet[idx,2])

gmresCtr = gmresCounter()
solVec, info = spla.gmres(conditionedMtrx, rhsVec, tol = gmresTol, callback = gmresCtr)
print "\nNumber of iterations for GMRES: "
print '\tnumIt =', gmresCtr.numIter
print ""



