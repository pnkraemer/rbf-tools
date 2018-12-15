# NAME: 'ptSets.py'
#
# PURPOSE: Collection of different strategies to construct pointsets
#
# DESCRIPTION: see PURPOSE; so far only the Euclidean case
#
# AUTHOR: NK, kraemer(at)ins.uni-bonn.de

import numpy as np
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


