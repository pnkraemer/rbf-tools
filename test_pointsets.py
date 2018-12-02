############################################
# NAME:
# test_pointsets.py

# PURPOSE:
# Test the different pointset generating functions 
# from pointset.py

# DESCRIPTION:
# We construct pointsets and plot them to assess the correctness

# AUTHOR: 
# Nicholas Kraemer, kraemer@ins.uni-bonn.de
############################################

import matplotlib.pyplot as plt
from pointsets import getptsTensorgrid, getptsRandomshiftlattice

print "\nHow many points per dimension?"
pointsperdim = input("\tEnter: ")

print "\nWe construct pointsets in dimension \n\td = 2\n"
dim = 2
totalnumber = pointsperdim**dim
print "Total number of points: \n\tN = ", '{:.0e}'.format(totalnumber), "\n"


ptsetTensorgrid = getptsTensorgrid(pointsperdim, dim)
plt.figure()
plt.plot(ptsetTensorgrid[:,0], ptsetTensorgrid[:,1], 'o', color = "darkslategray")
plt.title("Tensorgrid")
plt.show()

ptsetLattice = getptsRandomshiftlattice(totalnumber, dim)
plt.figure()
plt.plot(ptsetLattice[:,0], ptsetLattice[:,1], 'o', color = "darkslategray")
plt.title("Randomly Shifted Lattice")
plt.show()






