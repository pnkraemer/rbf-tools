# NAME: 'lebConstImqKernel.py'

# PURPOSE: Approximate Lebesgue constant for IMQ kernel

# DESCRIPTION: Compute and plot approximation on [0,1]

# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.insert(0,'../')
from kernelFcts import imqKernel
from kernelMtrcs import buildKernelMtrx

from functools import partial

np.random.seed(15051994)
np.set_printoptions(precision = 1)
plt.rcParams.update({'font.size': 16})

print "\nHow many points shall we work with? (e.g. 10)"
print "\tnumPts = ?"
numPts = input("Enter: ")

print "\nWhich power for the IMQ function? (e.g. 2.0)"
print "\tmaternReg = ?"
imqPower = 1.0*input("Enter: ")
print ""

imqKernelFixedPwr = partial(imqKernel, imqPower = imqPower)

ptSet = np.zeros((numPts, 1))
ptSet[:,0] = np.linspace(0,1,numPts)

kernelMtrx = buildKernelMtrx(ptSet, ptSet, imqKernelFixedPwr)

invKernelMtrx = np.linalg.inv(kernelMtrx)

numEvalPts = 400
evalPtSet = np.zeros((numEvalPts, 1))
evalPtSet[:,0] = np.linspace(0,1,numEvalPts)

kernelMtrxLeft = buildKernelMtrx(evalPtSet, ptSet, imqKernelFixedPwr)
lagFcts = kernelMtrxLeft.dot(invKernelMtrx)

lebConst = np.amax(lagFcts.sum(axis = 1))


randIdx = np.random.randint(numPts)
plt.plot(evalPtSet, lagFcts[0:numEvalPts,randIdx], linewidth = 3, label = "Lag. fct.", color = "darkslategray")
plt.plot(evalPtSet, lebConst * np.ones(numEvalPts), linewidth = 3, color = 'red', label = "Leb. const.")

plt.title("Lebesgue const. & randomly chosen Lag. fct.")
plt.legend(shadow=True)
plt.grid()
plt.savefig("figures/lebConstImqKernel/lebConst_%i_%.1f.png"%(numPts, imqPower))
plt.show()




