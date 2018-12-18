# NAME: 'ptSets.py'
#
# PURPOSE: Collection of different strategies to construct pointsets
#
# DESCRIPTION: see PURPOSE; all spherical functions act on S^2 \subseteq \R^3
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
from kernelFcts import distSphere 
from kernelMtrcs import buildKernelMtrx


def getPtsRandomSphere(size):
	ptsEucl = np.random.rand(size, 3)
	ptsSphere = np.zeros((size, 3))
	for idx in range(size):
		ptsSphere[idx,:] = ptsEucl[idx,:]/np.linalg.norm(ptsEucl[idx,:])
	return ptsSphere

# stolen from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def getPtsFibonacciSphere(samples=1,randomize=False):
    rnd = 1.0
    if randomize:
        rnd = np.random.rand() * samples
    points = []
    offset = 2.0/samples
    increment = np.pi * (3.0 - np.sqrt(5.0));
    for idx in range(samples):
        y = ((idx * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - y**2)
        phi = ((idx + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x,y,z])
    return np.array(points)



# stolen from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
def getPtsHalton(size, dim):
	def nextPrime():
		def isPrime(num):
			for i in range(2,int(num**0.5)+1):
				if(num % i)==0: return False
			return True
		prime = 3
		while(1):
			if isPrime(prime):
				yield prime
			prime += 2
	def vanDerCorput(n, base=2):
		vdc, denom = 0, 1
		while n:
			denom *= base
			n, remainder = divmod(n, base)
			vdc += remainder/float(denom)
		return vdc
	seq = []
	primeGen = nextPrime()
	next(primeGen)
	for d in range(dim):
		base = next(primeGen)
		seq.append([vanDerCorput(i, base) for i in range(size)])
	return np.array(seq).T

















