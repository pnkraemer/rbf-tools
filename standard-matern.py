# This program checks whether the Lagrange functions used in 
# approximation a divergence-free vector field exhibit an exponential decay
#
# A source: "DIVERGENCE-FREE KERNEL METHODS FOR APPROXIMATING THE STOKES PROBLEM", H. Wendland
# URL: https://epubs.siam.org/doi/pdf/10.1137/080730299
#
# September 26, 2018 
# kraemer@ins.uni-bonn.de
# 



import sympy
import numpy
import matplotlib.pyplot
from halton import halton_sequence


numpy.set_printoptions(precision = 1)

# Determine size of the checkup
print "\nHow many points shall we work with?(>25)"
NUMBER = input("Enter: ")

# Determine which plots to show
print "\nWanna see the chosen evaluation point?"
print "\tYes = 1\n\t No = 0"
NUMBER1 = input("Enter: ")

print "\nWanna see the heatmap of the condensed inverse kernel?"
print "\tYes = 1\n\t No = 0"
NUMBER2 = input("Enter: ")

print "\nWanna see the heatmap of the condensed inverse kernel,"
print "where the rows and columns are ordered by distance?"
print "\tYes = 1\n\t No = 0"
NUMBER3 = input("Enter: ")

print "\nWanna see a plot of these entries?"
print "\tYes = 1\n\t No = 0"
NUMBER5 = input("Enter: ")


print "\nWanna see the decay of the Lagrange function coefficients?"
print "\tYes = 1\n\t No = 0"
NUMBER4 = input("Enter: ")







# Function to build a matrix for divergence-free kernels
def buildkernel(X,Y):
	XX = numpy.zeros((len(X),len(Y)))
	for i in range(len(X)):
		for j in range(len(Y)):
			dummy = X[i] - Y[j]
			dummy2 = numpy.linalg.norm(dummy)
			XX[i,j] = MMfunct1(dummy[0], dummy[1])
	return XX



# Pick a basis function
#wendlandfunct = (1-sympy.Symbol('r'))**10 * (429 * sympy.Symbol('r')**4 +\
#	450 * sympy.Symbol('r')**3 + 210*sympy.Symbol('r')**2 + 50*sympy.Symbol('r') + 5)
sigma = 1.0
rho = 1.0
matern1 = sigma**2 * sympy.exp(-sympy.Symbol('a') / rho)
matern2 = sigma**2 * (1 + numpy.sqrt(3) * sympy.Symbol('a') / rho) *\
	sympy.exp(-numpy.sqrt(3) * sympy.Symbol('a') / rho)
matern3 = sigma**2 * (1 + numpy.sqrt(5) * sympy.Symbol('a') / rho +\
	5 * sympy.Symbol('a')**2 / (3 *rho**2)) *\
	sympy.exp(-numpy.sqrt(5) * sympy.Symbol('a') / rho)


MATERNFUNCTION = matern2
# Translate basis functions into our divergence-free setting (matrices)
R = sympy.sqrt(sympy.Symbol('x')**2 + sympy.Symbol('y')**2)
MM = MATERNFUNCTION.subs({'a': R})
MMfunct1 = sympy.lambdify((sympy.Symbol('x'), sympy.Symbol('y')), MM, modules = ['sympy', 'numpy'])

# Build pointset (Halton)
X = halton_sequence(NUMBER + 1,2)
X = X[1:,:]

# Build divergence-free kernel matrix
KERNEL = buildkernel(X,X)



# Construct Lagrange functions, use random evaluation points
KERNinv = numpy.linalg.inv(KERNEL)
Z = numpy.random.rand(NUMBER, 2)
G = buildkernel(Z,X)
T = G.dot(KERNinv)

# Check 'sizes' of the Lagrange function (matrices)
norming = numpy.zeros(NUMBER)
distance = 0 * norming
distance2 = 0 * norming
for i in range(NUMBER):
	DUM = T[i, 16]
	norming[i] = numpy.linalg.norm(DUM)
	#norming[i] = numpy.lingalg(T[2*i]**2 + T[2*i+1]**2)
	distance[i] = numpy.linalg.norm(Z[i,:] - X[16,:])
	distance2[i] = numpy.linalg.norm(X[i,:] - X[16,:])
#print distance2
#print distance
#print norming





# Analyse the 'decay' of the Lagrange function 
# coefficients in the inverse kernel matrix
CCCCC = numpy.zeros((NUMBER, NUMBER))
for i in range(len(CCCCC)):
	for j in range(len(CCCCC.T)):
		CCC = KERNinv[i,j]
		CCCCC[i,j] = numpy.linalg.norm(CCC)

# pick 17th entry and sort row/column by distances
mtrxx = numpy.zeros((len(CCCCC), len(CCCCC)))
mtrxx[16,:] = CCCCC[16,:]
mtrxx[:,16] = CCCCC[:,16]
i = numpy.argsort(distance2)
mtrxx[16,:] = mtrxx[16,i]
mtrxx[:,16] = mtrxx[16,:].T




# Plot the selected (16th) evaluation point
if NUMBER1 == 1:
	matplotlib.pyplot.plot(X[:,0], X[:,1], 'o', color = 'blue', markersize = 6)
	matplotlib.pyplot.plot(X[16,0], X[16,1], 'o', color = 'red', markersize = 6)
	matplotlib.pyplot.xlim(-0.2, 1.2)
	matplotlib.pyplot.ylim(-0.2, 1.2)
	matplotlib.pyplot.title('Which point was selected')
	matplotlib.pyplot.show()



# Plot 'heatmap' of condensed inverse kernel
if NUMBER2 == 1:
	image = matplotlib.pyplot.imshow(CCCCC, cmap='Greens', interpolation='nearest')
	matplotlib.pyplot.colorbar(image)
	matplotlib.pyplot.title('Heatmap of condensed inverse kernel')
	matplotlib.pyplot.show()

# Plot 'heatmap' of row/column (sorted by distance)
if NUMBER3 == 1:
	image2 = matplotlib.pyplot.imshow(mtrxx, cmap='Greens', interpolation='nearest')
	matplotlib.pyplot.colorbar(image2)
	matplotlib.pyplot.title('Heatmap of condensed inverse kernel (selected and sorted)')
	matplotlib.pyplot.show()

# Plot decay of matrix coefficients
if NUMBER5 == 1:
	matplotlib.pyplot.semilogy(distance2[i], mtrxx[16,:], 'o')
	matplotlib.pyplot.grid()
	matplotlib.pyplot.xlabel('Distance from 16th point')
	matplotlib.pyplot.ylabel('Entries of inverse kernel matrix')
	matplotlib.pyplot.title('Decay of inverse Kernel matrix coefficients (norms of matrices)')
	matplotlib.pyplot.show()




# Plot norm of the Lagrange function matrices 
# corresponding to the distance to the selected point
if NUMBER4 == 1:
	matplotlib.pyplot.semilogy(distance, norming, 'o')
	matplotlib.pyplot.grid()
	matplotlib.pyplot.xlabel('Distance from 16th point')
	matplotlib.pyplot.ylabel('Absolute value of Lagrange function evaluations')
	matplotlib.pyplot.title('Decay of Lagrange function "values" (norms of matrices)')
	matplotlib.pyplot.show()






# ideas:
# try higher order matern kernel (done)

# pick a single row/column of inverse kernel matrix 
# and look at the decay (done)

# try larger pointsets

# check whether the normal matern kernel
# exhibits this decay at this pointsetsize
# use more efficient evaluation of derivatives of 
# matern kernels without sympy


