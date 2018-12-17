 #############################################################################
# 
# This script checks whether the local-Lagrange function preconditioner works 
# with symmetric (higher order) thin-plate spline collocation.
#
# Nicholas Kraemer
# kraemer@ins.uni-bonn.de
#
# Last changed: July 20, 2018
# 
#############################################################################

import numpy as np
import scipy.io as sp
import sys
import sympy

from sympy.abc import x,y,z, r, p, t, s	# never use for anything else now!!!
from sympy import refine, Q, sqrt

import matplotlib.pyplot as plt

import scipy.spatial
import scipy.sparse.linalg as spla




class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

np.set_printoptions(precision=1)



# Get pointset
def get_pde_param():

	# Pick a set
	print "\nWhich PDE parameter?"
	print "    1 = 1.0, 2 = 2.0, 3 = 10.0"
	idx_number_collpoints = input("Enter: ")
	if idx_number_collpoints == 1:
 		mat_collpoints = 1.0
	elif idx_number_collpoints == 2:
 		mat_collpoints = 2.0
	elif idx_number_collpoints == 3:
 		mat_collpoints = 10.0
	return mat_collpoints


# Computes and returns the differential operator of a function using sympy
def compute_differential_operator_cartesian(EXPR_FCT, PARAM_PDE):
	
	# Spherical Coordinates
	EXPR_X = sympy.sin(t) * sympy.cos(p)
	EXPR_Y = sympy.sin(t) * sympy.sin(p)
	EXPR_Z = sympy.cos(t)
	EXPR_FCT_SPHERICAL = EXPR_FCT.subs({'x': EXPR_X, 'y': EXPR_Y, 'z': EXPR_Z})

	# Laplace Beltrami Operator
	EXPR_LPB_OF_FCT =  sympy.diff(sympy.sin(t) * sympy.diff(EXPR_FCT_SPHERICAL,t), \
		t) / (sympy.sin(t)) + sympy.diff(EXPR_FCT_SPHERICAL, p, p) / (sympy.sin(t)**2)

	# Back to Cartesian Coordinates
	EXPR_T = sympy.acos(z)
	EXPR_P = sympy.atan2(y, x)
	EXPR_LPB_CARTESIAN = EXPR_LPB_OF_FCT.subs({'t': EXPR_T, 'p': EXPR_P})


	# Full operator
	EXPR_FULL_OP = -EXPR_LPB_CARTESIAN + PARAM_PDE**2 * EXPR_FCT
	return sympy.simplify(EXPR_FULL_OP)



# Define spherical harmonics
EXPR_SPH_00 = 0 *sympy.sin(x) + 1
EXPR_SPH_10 = 0 *sympy.sin(x) + x
EXPR_SPH_11 = 0 *sympy.sin(x) + y
EXPR_SPH_12 = 0 *sympy.sin(x) + z
EXPR_SPH_20 = 0 *sympy.sin(x) + x*y
EXPR_SPH_21 = 0 *sympy.sin(x) + y*z
EXPR_SPH_22 = 2*z**2 - x**2 - y**2
EXPR_SPH_23 = x*z
EXPR_SPH_24 = x**2 - y**2
EXPR_SPH_30 = (3*x**2 - y**2) * y
EXPR_SPH_31 = x*y*z
EXPR_SPH_32 = y*(4*z**2 - x**2 - y**2)
EXPR_SPH_33 = z*(2*z**2 - 3*x**2 - 3*y**2)
EXPR_SPH_34 = x*(4*z**2 - x**2 - y**2)
EXPR_SPH_35 = (x**2 - y**2)*z
EXPR_SPH_36 = (x**2 - 3*y**2)*x

SPH_00 = sympy.lambdify((x,y,z), EXPR_SPH_00, modules=['numpy', 'sympy']) 
SPH_10 = sympy.lambdify((x,y,z), EXPR_SPH_10, modules=['numpy', 'sympy']) 
SPH_11 = sympy.lambdify((x,y,z), EXPR_SPH_11, modules=['numpy', 'sympy']) 
SPH_12 = sympy.lambdify((x,y,z), EXPR_SPH_12, modules=['numpy', 'sympy']) 
SPH_20 = sympy.lambdify((x,y,z), EXPR_SPH_20, modules=['numpy', 'sympy']) 
SPH_21 = sympy.lambdify((x,y,z), EXPR_SPH_21, modules=['numpy', 'sympy']) 
SPH_22 = sympy.lambdify((x,y,z), EXPR_SPH_22, modules=['numpy', 'sympy']) 
SPH_23 = sympy.lambdify((x,y,z), EXPR_SPH_23, modules=['numpy', 'sympy']) 
SPH_24 = sympy.lambdify((x,y,z), EXPR_SPH_24, modules=['numpy', 'sympy']) 
SPH_30 = sympy.lambdify((x,y,z), EXPR_SPH_30, modules=['numpy', 'sympy']) 
SPH_31 = sympy.lambdify((x,y,z), EXPR_SPH_31, modules=['numpy', 'sympy']) 
SPH_32 = sympy.lambdify((x,y,z), EXPR_SPH_32, modules=['numpy', 'sympy']) 
SPH_33 = sympy.lambdify((x,y,z), EXPR_SPH_33, modules=['numpy', 'sympy']) 
SPH_34 = sympy.lambdify((x,y,z), EXPR_SPH_34, modules=['numpy', 'sympy']) 
SPH_35 = sympy.lambdify((x,y,z), EXPR_SPH_35, modules=['numpy', 'sympy']) 
SPH_36 = sympy.lambdify((x,y,z), EXPR_SPH_36, modules=['numpy', 'sympy']) 




# Get pointset
def get_pointset_halton():

	# Pick a set
	print "\nWhich pointset for COLLOCATION?"
	print "    1 = 151, 2 = 501, 3 = 1251, 4 = 5001, 5 = 10001, 6 = 25001"
	idx_number_collpoints = input("Enter: ")
	if idx_number_collpoints == 1:
 		mat_collpoints = sp.loadmat('../spherepoints/151points.mat')
	elif idx_number_collpoints == 2:
 		mat_collpoints = sp.loadmat('../spherepoints/501points.mat')
	elif idx_number_collpoints == 3:
 		mat_collpoints = sp.loadmat('../spherepoints/1251points.mat')
	elif idx_number_collpoints == 4:
 		mat_collpoints = sp.loadmat('../spherepoints/5001points.mat')
	elif idx_number_collpoints == 5:
 		mat_collpoints = sp.loadmat('../spherepoints/10001points.mat')
	elif idx_number_collpoints == 6:
 		mat_collpoints = sp.loadmat('../spherepoints/25001points.mat')
	else:
 		mat_collpoints = sp.loadmat('../spherepoints/501points.mat')

 	# Get pointset (in cart. coord.)
	collpoints0 = mat_collpoints['X']

	return collpoints0


def get_pointset_random():
	print "\nHow many points for EVALUATION? "
	print "    1 = 50, 2 = 200, 3 = 500 , 4 = 1000, 5 = 3000, 6 = 7500"
	idx_number_evalpoints = input("Enter: ")
	if idx_number_evalpoints == 1:
 		len_evalpoints = 50
	elif idx_number_evalpoints == 2:
 		len_evalpoints = 200
	elif idx_number_evalpoints == 3:
 		len_evalpoints = 500
	elif idx_number_evalpoints == 4:
 		len_evalpoints = 1000
	elif idx_number_evalpoints == 5:
 		len_evalpoints = 3000
	elif idx_number_evalpoints == 6:
 		len_evalpoints = 7500

	np.random.seed(1)
	evalpoints0 = np.random.rand(len_evalpoints,3)
	evalpoints = np.zeros((len(evalpoints0),2))
	for i in range(len_evalpoints):
		evalpoints0[i,:] = 2 * evalpoints0[i,:] - 1
		evalpoints0[i,:] = evalpoints0[i,:]/np.linalg.norm(evalpoints0[i,:])

	return evalpoints0


def get_constant():
	print "\n|    Which constant for the radius? "
	print "|        1 = 5.0, 2 = 10.0, 3 = 15.0 , 4 = 25.0, 5 = 100.0"
	idx_const = input("|    Enter: ")
	if idx_const == 1:
 		constant = 5.0
	elif idx_const == 2:
 		constant = 10.0
	elif idx_const == 3:
 		constant = 15.0
	elif idx_const == 4:
 		constant = 25.0
	elif idx_const == 5:
 		constant = 100.0
 	return constant


def get_tolerance():
	print "|\n|    Which tolerance for GMRES? "
	print "|        1 = 1e-6, 2 = 1e-8, 3 = 1e-12, 4 = 1e-16"
	idx_const = input("|    Enter: ")
	if idx_const == 1:
 		constant = 1e-6
	elif idx_const == 2:
 		constant = 1e-8
	elif idx_const == 3:
 		constant = 1e-12
	elif idx_const == 4:
 		constant = 1e-16

 	return constant

# Assembles rhs vector using pointset and expression of rhs
def assemble_rhs(PSET, EXPR_RHS):

	# Turn expression into function
	EVALUATE_RHS = sympy.lambdify((x,y,z), EXPR_RHS, modules=['numpy', 'sympy']) 

	# Assemble rhs-array
	VEC_RHS = np.zeros(len(PSET) + 16)
	for R in range(len(PSET)):
		VEC_RHS[R] = EVALUATE_RHS(PSET[R,0], PSET[R,1], PSET[R,2])
	return VEC_RHS







# Get approximate fill distance
def get_filldistance(PSET):

	# Get reference points - random because Halton pointset is nested
	np.random.seed(2)
	amount = 1111
	evalpoints0 = np.random.rand(amount,3)
	Y = np.zeros((len(evalpoints0),2))
	for i in range(amount):
		evalpoints0[i,:] = 2 * evalpoints0[i,:] - 1
		evalpoints0[i,:] = evalpoints0[i,:]/np.linalg.norm(evalpoints0[i,:])

	Y = evalpoints0
	# Construct Gram-type matrix
	MTRXX = np.zeros((len(PSET), len(Y)))
	MTRXX = PSET.dot(Y.T)
	for R in range(len(PSET)):
		for C in range(len(Y)):
			if MTRXX[R,C]>= 1:
				MTRXX[R,C] = 0
			else:
				MTRXX[R,C] = np.sqrt(2-2*MTRXX[R,C])
	
	# Return fill distance
	return np.amax(MTRXX.min(axis = 0))





# Assembles the kernelmatrix using given rbf
def assemble_collmatrix_tps(PSET, QSET, EXPR_RBF, PDE_PARAM):


	# Differentiate RBF
	HELP_1 = sympy.diff(EXPR_RBF, s)
	HELP_2 = sympy.diff(HELP_1 * (1-s*s), s)
	EXPR_DIFF_RBF = sympy.simplify(HELP_2)
	
	HELP_3 = sympy.diff(EXPR_DIFF_RBF, s)
	HELP_4 = sympy.diff(HELP_3 * (1-s*s), s)
	EXPR_DIFF_DIFF_RBF = sympy.simplify(HELP_4)

	# Turn expressions into functions
	EVALUATE_RBF = sympy.lambdify(s, EXPR_RBF, modules=['numpy', 'sympy'])
	EVALUATE_DIFF_RBF = sympy.lambdify(s, EXPR_DIFF_RBF, modules=['numpy', 'sympy']) 
	EVALUATE_DIFF_DIFF_RBF = sympy.lambdify(s, EXPR_DIFF_DIFF_RBF, modules=['numpy', 'sympy']) 

	# Construct matrix
	NUM_PTS = len(PSET)
	NUM_QTS = len(QSET)
	MTRXX = np.zeros((NUM_PTS + 16, NUM_QTS + 16))
	MTRXX[0:NUM_PTS, 0:NUM_QTS] = PSET.dot(QSET.T)
	for R in range(len(PSET)):
		for C in range(len(QSET)):
			if MTRXX[R,C]>=1:
				MTRXX[R,C] = 0
			else:
				MTRXX[R,C] = EVALUATE_DIFF_DIFF_RBF(MTRXX[R,C]) - 2 * PDE_PARAM**2 * EVALUATE_DIFF_RBF(MTRXX[R,C]) + PDE_PARAM**4 * EVALUATE_RBF(MTRXX[R,C])

	# Fake the Lap-Bel Operator
	l1 = 0 + PDE_PARAM**2
	l2 = 2 + PDE_PARAM**2
	l3 = 6 + PDE_PARAM**2
	l4 = 12 + PDE_PARAM**2

	# Collocated polynomial part right
	for i in range(NUM_PTS):
		MTRXX[i, NUM_QTS] = l1 * SPH_00(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 1] = l2 * SPH_10(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 2] = l2 * SPH_11(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 3] = l2 * SPH_12(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 4] = l3 * SPH_20(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 5] = l3 * SPH_21(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 6] = l3 * SPH_22(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 7] = l3 * SPH_23(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 8] = l3 * SPH_24(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 9] = l4 * SPH_30(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 10] = l4 * SPH_31(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 11] = l4 * SPH_32(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 12] = l4 * SPH_33(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 13] = l4 * SPH_34(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 14] = l4 * SPH_35(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 15] = l4 * SPH_36(PSET[i,0], PSET[i,1], PSET[i,2])

	# Collocated polynomial part bottom
	for i in range(NUM_QTS):
		MTRXX[NUM_PTS, i] = l1 * SPH_00(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 1, i] = l2 * SPH_10(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 2, i] = l2 * SPH_11(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 3, i] = l2 * SPH_12(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 4, i] = l3 * SPH_20(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 5, i] = l3 * SPH_21(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 6, i] = l3 * SPH_22(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 7, i] = l3 * SPH_23(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 8, i] = l3 * SPH_24(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 9, i] = l4 * SPH_30(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 10, i] = l4 * SPH_31(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 11, i] = l4 * SPH_32(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 12, i] = l4 * SPH_33(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 13, i] = l4 * SPH_34(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 14, i] = l4 * SPH_35(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 15, i] = l4 * SPH_36(QSET[i,0], QSET[i,1], QSET[i,2])
	return MTRXX





# Assembles the kernelmatrix using given rbf
def assemble_approx_for_coll(PSET, QSET, EXPR_RBF, PDE_PARAM):


	# Differentiate RBF
	HELP_1 = sympy.diff(EXPR_RBF, s)
	HELP_2 = sympy.diff(HELP_1 * (1-s*s), s)
	EXPR_DIFF_RBF = sympy.simplify(HELP_2)

	# Turn expressions into functions
	EVALUATE_RBF = sympy.lambdify(s, EXPR_RBF, modules=['numpy', 'sympy'])
	EVALUATE_DIFF_RBF = sympy.lambdify(s, EXPR_DIFF_RBF, modules=['numpy', 'sympy']) 

	# Construct matrix
	NUM_PTS = len(PSET)
	NUM_QTS = len(QSET)
	MTRXX = np.zeros((NUM_PTS + 16, NUM_QTS + 16))
	MTRXX[0:NUM_PTS, 0:NUM_QTS] = PSET.dot(QSET.T)
	for R in range(len(PSET)):
		for C in range(len(QSET)):
			if MTRXX[R,C]>=1:
				MTRXX[R,C] = 0
			else:
				MTRXX[R,C] = - EVALUATE_DIFF_RBF(MTRXX[R,C]) + PDE_PARAM**2 * EVALUATE_RBF(MTRXX[R,C])

	# Collocated polynomial part right
	for i in range(NUM_PTS):
		MTRXX[i, NUM_QTS] = SPH_00(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 1] = SPH_10(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 2] = SPH_11(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 3] = SPH_12(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 4] = SPH_20(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 5] = SPH_21(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 6] = SPH_22(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 7] = SPH_23(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 8] = SPH_24(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 9] = SPH_30(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 10] = SPH_31(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 11] = SPH_32(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 12] = SPH_33(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 13] = SPH_34(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 14] = SPH_35(PSET[i,0], PSET[i,1], PSET[i,2])
		MTRXX[i, NUM_QTS + 15] = SPH_36(PSET[i,0], PSET[i,1], PSET[i,2])

	# Collocated polynomial part bottom
	for i in range(NUM_QTS):
		MTRXX[NUM_PTS, i] = SPH_00(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 1, i] = SPH_10(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 2, i] = SPH_11(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 3, i] = SPH_12(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 4, i] = SPH_20(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 5, i] = SPH_21(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 6, i] = SPH_22(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 7, i] = SPH_23(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 8, i] = SPH_24(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 9, i] = SPH_30(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 10, i] = SPH_31(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 11, i] = SPH_32(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 12, i] = SPH_33(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 13, i] = SPH_34(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 14, i] = SPH_35(QSET[i,0], QSET[i,1], QSET[i,2])
		MTRXX[NUM_PTS + 15, i] = SPH_36(QSET[i,0], QSET[i,1], QSET[i,2])
	return MTRXX







def local_lagrange_coll(PSET, TREE, CONST, EXPR_RBF, PDE_PARAM):

	LOCLAN = np.zeros((len(PSET) + 16, len(PSET)))

	# determine number of neighbors
	N = len(PSET)
	n = CONST * np.log10(N) * np.log10(N)
	n = np.minimum(np.floor(n), N)
	print "|\n|    Number of neighbors for localized Lagrange functions:"
	print "|        n =", n


	for P in range(len(PSET)):

		ds, ind = TREE.query(PSET[P], k = n)
		MTRX = assemble_collmatrix_tps(PSET[ind], PSET[ind], EXPR_RBF, PDE_PARAM)

		RHSLOC = np.zeros(len(ind) + 16)
		RHSLOC[(ind==P*np.ones((1, len(ind)))[0]).nonzero()] = 1
		localcoeff = np.linalg.solve(MTRX, RHSLOC)

		coeff = localcoeff[range(len(ind))]
		poly = localcoeff[range(len(ind), len(ind) + 16)]

		LOCLAN[ind, P] = coeff.T
		LOCLAN[len(PSET):(len(PSET)+16), P] = poly.T


	return LOCLAN

























#############################################################################
# MAIN BLOCK
#############################################################################





# Define the function to be differentiated and the parameters for the differential operator
#expr_solution = 0.75 * sympy.exp(-0.25 * (9*x-2)**2 - 0.25 * (9*y-2)**2) \
#	+ 0.75 * sympy.exp(- 1/49 * (9*x+1)**2 - 0.1 * (9*y+1)) \
#	+ 0.5 * sympy.exp(-0.25 * (9*x-7)**2 - 0.25 * (9*y-3)**2) \
#	- 0.2 * sympy.exp(-(9*x-4)**2 - (9*y-7)**2)
expr_solution = sympy.exp(x) + (1-x)
evaluate_solution = sympy.lambdify((x,y,z), expr_solution, modules=['numpy', 'sympy']) 
pde_param = get_pde_param()

# Define RBF
expr_rbf_t = (1-s)**3 * sympy.log(1-s)

# Get pointsets
collpoints = get_pointset_halton()
evalpoints = get_pointset_random()

# Check remaining parameters
print "\nPreconditioner? "
print "    1 = YES, 2 = NO"
truth = input("Enter: ")
if truth == 1:
	const = get_constant()
	tolerance = get_tolerance()

	print "|\n|    Compare eigenvalues as well?"
	print "|        1 = YES, 2 = NO"
	eigentruth = input("|    Enter: ")

# Build KD-Tree
tree = scipy.spatial.KDTree(collpoints)


# Assemble Kernelmatrix
L2 = assemble_collmatrix_tps(collpoints, collpoints, expr_rbf_t, pde_param)

# Assemble RHS
expr_rhs = compute_differential_operator_cartesian(expr_solution, pde_param)
vec_rhs = assemble_rhs(collpoints, expr_rhs)


if truth == 1:
	global_precon = local_lagrange_coll(collpoints, tree, const, expr_rbf_t, pde_param)
	LL =  L2.dot(global_precon)

	LL = LL[0:len(collpoints), 0:len(collpoints)]
	vec_rhs = vec_rhs[range(len(collpoints))]
	
	counter = gmres_counter()
	vec_solution, info = spla.gmres(LL, vec_rhs, tol = tolerance, callback = counter)
	print "|\n|    Number of iterations for GMRES: "
	print '|        #It =', counter.niter
	vec_solution = global_precon.dot(vec_solution)

	# Check eigenvalues
	if eigentruth == 1:
		A1 = np.linalg.eig(L2)[0]
		A2 = np.linalg.eig(LL)[0]

		plt.semilogy(np.absolute(A1), ".", label = 'K (no precond.)', color = 'k', markersize = 10)
		plt.semilogy(np.absolute(A2), ".", label = 'KP (with precond.)', color = 'r', markersize = 10)
		plt.grid()
		plt.legend()
		plt.xlim(-10, len(collpoints) + 10)
		plt.title("Absolute values of the eigenvalues")
		plt.show()

else:
	vec_solution = np.linalg.solve(L2, vec_rhs)

h1 = get_filldistance(collpoints)
h2 = get_filldistance(evalpoints)
print '\nApproximate fill distance of:'
print '    collpoints: h =', h1
print '    evalpoints: h =', h2

# Construct approximate solution
mtrx_kernel = assemble_approx_for_coll(evalpoints, collpoints, expr_rbf_t, pde_param)
vec_sol_approx = mtrx_kernel.dot(vec_solution)
vec_sol_approx = vec_sol_approx[0:len(evalpoints)]


# Assemble optimal solution
vec_sol_reference = np.zeros(len(evalpoints))
for i in range(len(evalpoints)):
	vec_sol_reference[i] =  evaluate_solution(evalpoints[i,0], evalpoints[i,1], evalpoints[i,2])

# Success of preconditioner
if truth == 1:
	print '\nEffect of preconditioner:'
	print '    cond(K) =', np.linalg.cond(L2)
	print '    cond(KP) =', np.linalg.cond(LL)
else:
	print '\nCondition number:'
	print '    cond(K) =', np.linalg.cond(L2)

# Print approximation errors
coll_err_l2 = np.linalg.norm(vec_sol_reference - vec_sol_approx) / np.sqrt(len(evalpoints))
coll_err_linf = np.amax(np.fabs(vec_sol_reference - vec_sol_approx))
print '\nCollocation error:'
print '    ||u - u_coll||_2/sqrt(M) =', coll_err_l2
print '    ||u - u_coll||_infty =', coll_err_linf, '\n'






























