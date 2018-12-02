############################################
# NAME:
# pointsets.py
#
# PURPOSE:
# Provide a collection of functions 
# which generate pointsets on different spaces
#
#
# DESCRIPTION:
# We include deterministic (grids, low-discrepancy series) and 
# random pointsets on Euclidean spaces (all on unit square/cube/...) 
# and spheres.
# 
#
# AUTHOR: 
# Nicholas Kraemer, kraemer@ins.uni-bonn.de
#
############################################

from __future__ import division
import numpy as np


# construct uniform tensor grid with N^dim points in dim dimensions
# N is number of gridpoints per dimension
def getptsTensorgrid(numberperdim, dim):
	grid1D = np.zeros((numberperdim, dim))
	for i in range(dim):
		grid1D[:,i] = np.linspace(-1,1,numberperdim)
	if dim > 1:
		tensormeshgrid = np.meshgrid(*grid1D.T)
		gridDimD = np.zeros((numberperdim**dim, dim))
		for idx in range(dim):
			gridDimD[:,idx] = np.array(tensormeshgrid[idx]).flatten()
	elif dim <= 1:
		gridDimD = grid1D
	return gridDimD


# Computes the lattice points for qmc integration
def getptsRandomshiftlattice(totalnumber, dim):

	# load generating vector
	genvec = np.loadtxt('/Users/nicholaskramer/Documents/GitHub/rbf-tools/pointsets/vec.txt')
	genvec = genvec[0:dim, 1]

	# generate lattice points and random shift
	lattice = np.zeros((totalnumber, dim))
	shift = np.random.rand(dim)
	for idx in range(totalnumber):
		lattice[idx,:] = (genvec * idx / totalnumber  + shift) % 1

	return lattice



