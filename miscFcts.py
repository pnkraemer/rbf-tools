# NAME: 'miscFcts.py'
#
# PURPOSE: Collect all support functions that do not fit the other categories
#
# DESCRIPTION: see PURPOSE; we collect e.g. sympy-based functions, fill distances, Lebesgue constants
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import sympy
from kernelFcts import distSphere

def computeLapBelOp(exprFct, pdeParam):
	exprX = sympy.sin(sympy.Symbol('t')) * sympy.cos(sympy.Symbol('p'))
	exprY = sympy.sin(sympy.Symbol('t')) * sympy.sin(sympy.Symbol('p'))
	exprZ = sympy.cos(sympy.Symbol('t'))
	exprFctPol = exprFct.subs({'x': exprX, 'y': exprY, 'z': exprZ})

	lapBelOpFctPol = sympy.diff(sympy.sin(sympy.Symbol('t')) * sympy.diff(exprFctPol,sympy.Symbol('t')), \
		sympy.Symbol('t')) / (sympy.sin(sympy.Symbol('t'))) +\
		sympy.diff(exprFctPol, sympy.Symbol('p'), sympy.Symbol('p')) / (sympy.sin(sympy.Symbol('t'))**2)

	exprT = sympy.acos(sympy.Symbol('z'))
	exprP = sympy.atan2(sympy.Symbol('y'), sympy.Symbol('x'))
	exprLapBelCart = lapBelOpFctPol.subs({'t': exprT, 'p': exprP})

	return -exprLapBelCart + pdeParam**2 * exprFct


def getApproxFillDistanceSphere(ptSet, numSamp = 1000, distFct = distSphere):
	evalPtSet = getPtsFibonacciSphere(numSamp, 1)
	distMtrx = buildKernelMtrx(ptSet, evalPtSet, distFct)
	return np.amax(distMtrx.min(axis = 0))









