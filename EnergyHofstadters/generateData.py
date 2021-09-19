import numpy as np
from mpmath import jtheta,sqrt, exp, pi, conj
from scipy.sparse.linalg import expm
from scipy.linalg import eigvalsh
from math import gcd
import sys

# parse the inputs.  inputString is the string that is fed to stdin
# reads python3 generateData.py 3 7, for example.
# inputString[0] = generateData.py
# inputString[1] = p
# inputString[2] = q

inputString = sys.argv
p = int(inputString[1])
q = int(inputString[2])

# magic angle = 1.05 degrees.  w0 and w1 defined relative
# to vfkth.  To get other angles scale vfkth by sin(theta/2)
# and w0, w1 with the opposite factor.

phi = (2*np.pi*p)/q
w0 = 0.8/np.sqrt(3)
w1 = 1.0/np.sqrt(3)
vfkth = 110*np.sqrt(3)

# throws and error (and ends the program) if there is a simpler p, q
assert (gcd(p,q) == 1)

# calculates the Siegel theta function from Jacobis
def siegel(z1,z2):
    return(sqrt(2)*exp(-pi*z1*z1-pi*z2*z2+2j*pi*z1*z2)* \
    jtheta(3,pi*(1.0j*z1+z2),exp(-pi))*jtheta(3,pi*(z1+1.0j*z2),exp(-pi)))

# get the phase e^{\xi_q(k)}
def getPhase(kvec,qvec,latticeVec1,latticeVec2):
    k1 = np.dot(kvec,latticeVec1)
    k2 = np.dot(kvec,latticeVec2)
    q1 = np.dot(qvec,latticeVec1)
    q2 = np.dot(qvec,latticeVec2)

    qComplex = q1 + 1.0j*q2
    numer = exp(-qComplex*conj(qComplex)/(8*pi))*siegel((k1-qComplex/2)/(2*pi),(k2+1.0j*qComplex/2)/(2*pi))
    denom= sqrt(siegel(k1/(2*pi),k2/(2*pi))*siegel((k1-q1)/(2*pi),(k2-q2)/(2*pi)))
    return(complex(numer/denom))

# lattice and reciprocal lattice vectors
a1 = 2./3*np.array([-np.sqrt(3)/2, -1./2],dtype=complex);
a2 = 2./3*np.array([np.sqrt(3)/2, -1./2],dtype=complex);
b1 = np.array([-np.sqrt(3)/2, -3./2],dtype=complex);
b2 = np.array([np.sqrt(3)/2, -3./2],dtype=complex);

# modified vectors, scaled by p and q.  Enclose 2\pi flux
ta1 = 1.0/p*a1
ta2 = q*a2
tb1 = p*b1
tb2 = 1.0/q*b2

# omega, z1, z2, needed to construct the form factor via exponentiation
omega = ta1[0]*ta2[1] - ta1[1]*ta2[0]
z1 = 1/np.sqrt(omega)*(ta1[0]+1.0j*ta1[1])
z2 = 1/np.sqrt(omega)*(ta2[0]+1.0j*ta2[1])

# num_landau is the number of landau levels we want to keep (in the energy basis),
# this becomes 2*num_landau + 1.  num_landau_calc is the size of the matrix we
# exponentiate, and it is truncated

num_landau_calc = 400
num_landau = 100

# build the form factor marix in the b1 direction. gQ is \gamma_q
qvec = 2*np.pi*b1
gQ = np.dot(qvec,ta1)*np.conj(z2)-np.dot(qvec,ta2)*np.conj(z1)
gQbar = np.conj(gQ)

toExp = np.zeros((num_landau_calc,num_landau_calc),dtype=complex)

# matrix of a^\daggers, and a's, with appropriate phases
for m in range(num_landau_calc-1):
    toExp[m,m+1] = np.sqrt(m+1)*1.0j*gQ/np.sqrt(4*np.pi)
    toExp[m+1,m] = np.sqrt(m+1)*1.0j*gQbar/np.sqrt(4*np.pi)

formFactors1 = expm(toExp)

# same for the b2 direction
qvec = 2*np.pi*b2
gQ = np.dot(qvec,ta1)*np.conj(z2)-np.dot(qvec,ta2)*np.conj(z1)
gQbar = np.conj(gQ)

toExp2 = np.zeros((num_landau_calc,num_landau_calc),dtype=complex)

for m in range(num_landau_calc-1):
    toExp2[m,m+1] = np.sqrt(m+1)*1.0j*gQ/np.sqrt(4*np.pi)
    toExp2[m+1,m] = np.sqrt(m+1)*1.0j*gQbar/np.sqrt(4*np.pi)

formFactors2 = expm(toExp2)

# build the kinetic energy terms
energy = np.sqrt(phi/(2*np.pi))* np.sqrt((3*np.sqrt(3))/(2*np.pi))
kinetic_1 = np.zeros((2*num_landau+1,2*num_landau+1),dtype = complex) # layer 1
kinetic_2 = np.zeros((2*num_landau+1,2*num_landau+1),dtype = complex) # layer 2

for m in range(-num_landau,num_landau+1):
    kinetic_1[m][m] = energy * np.sqrt(abs(m)) *np.sign(m)
    kinetic_2[m][m] = energy * np.sqrt(abs(m)) *np.sign(m)

# And then the shifts from the change of basis
for m in range(-num_landau,num_landau+1):
    for n in range(-num_landau,num_landau+1):
        if (abs(m)-1 == abs(n)):
            coeff = 1.0/((np.sqrt(2))**(np.sign(abs(n)) + np.sign(abs(m))))
            kinetic_2[m][n] = 0.5j*np.sign(m)*coeff
            kinetic_1[m][n] = -0.5j*np.sign(m)*coeff
        elif (abs(m) == abs(n)-1):
            coeff = 1.0/((np.sqrt(2))**(np.sign(abs(n)) + np.sign(abs(m))))
            kinetic_2[m][n] = -0.5j*np.sign(n)*coeff
            kinetic_1[m][n] = 0.5j*np.sign(n)*coeff

# now the T-matrices
t1_matrix = np.zeros((2*num_landau+1,2*num_landau+1),dtype = complex)
t2_matrix = np.zeros((2*num_landau+1,2*num_landau+1),dtype = complex)
t3_matrix = np.zeros((2*num_landau+1,2*num_landau+1),dtype = complex)

# but in the energy basis, so it's a lot nastier
for m in range(-num_landau,num_landau+1):
    for n in range(-num_landau,num_landau+1):
        coe = 1.0/((np.sqrt(2))**(np.sign(abs(n)) + np.sign(abs(m))))

        t2_matrix[m][n] += coe*w0*formFactors1[abs(m),abs(n)]
        t3_matrix[m][n] += coe*w0*formFactors2[abs(m),abs(n)]

        if abs(n) > 0 and abs(m) > 0:
            t2_matrix[m][n] += np.sign(m*n)*coe*w0*formFactors1[abs(m)-1,abs(n)-1]
            t3_matrix[m][n] += np.sign(m*n)*coe*w0*formFactors2[abs(m)-1,abs(n)-1]

        if abs(n) > 0:
            t2_matrix[m][n] += np.sign(n)*coe*w1*np.exp(-2.0j*np.pi/3)*formFactors1[abs(m),abs(n)-1]
            t3_matrix[m][n] += np.sign(n)*coe*w1*np.exp(2.0j*np.pi/3)*formFactors2[abs(m),abs(n)-1]

        if abs(m) > 0:
            t2_matrix[m][n] += np.sign(m)*coe*w1*np.exp(2.0j*np.pi/3)*formFactors1[abs(m)-1,abs(n)]
            t3_matrix[m][n] += np.sign(m)*coe*w1*np.exp(-2.0j*np.pi/3)*formFactors2[abs(m)-1,abs(n)]

        if abs(m) == abs(n):
            t1_matrix[m][n] += w0*coe
            if abs(m) != 0:
                t1_matrix[m][n] += w0*coe*np.sign(m*n)

        if abs(m)-1 == abs(n) and abs(m) != 0:
            t1_matrix[m][n] += w1*coe*np.sign(m)

        if abs(n)-1 == abs(m) and abs(n) != 0:
            t1_matrix[m][n] += w1*coe*np.sign(n)

# block it all together into a big matrix
def generateHamiltonian(kvec):
    diagKinetic1 = [[np.zeros((2*num_landau+1,2*num_landau+1),dtype='complex') for v in range(p)] for u in range(p)]
    diagKinetic2 = [[np.zeros((2*num_landau+1,2*num_landau+1),dtype='complex') for v in range(p)] for u in range(p)]

    diagPotential = [[np.zeros((2*num_landau+1,2*num_landau+1),dtype='complex') for v in range(p)] for u in range(p)]
    offDiagPotential = [[np.zeros((2*num_landau+1,2*num_landau+1),dtype='complex') for v in range(p)] for u in range(p)]

    for v in range(p):
        diagKinetic1[v][v] = kinetic_1
        diagKinetic2[v][v] = kinetic_2

        diagPotential[v][v] = t1_matrix+getPhase(kvec+2*np.pi*b1*v,2*np.pi*b2,ta1,ta2)*t3_matrix
        # -1 entry is last entry
        offDiagPotential[v-1][v] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b1,ta1,ta2)*t2_matrix

    potentialTerm = np.block(diagPotential) + np.block(offDiagPotential)
    hamiltonian = np.block([[np.block(diagKinetic1),potentialTerm],[potentialTerm.conj().T,np.block(diagKinetic2)]])
    return(vfkth*hamiltonian)

# and now assemble a histogram.
histPoints = []
kPoints = []

# depending on q, we will restrict the number of points to consider in
# the k-space mesh.

if q > 10:
    rangeLimit = 1
elif q > 5:
    rangeLimit = 2
elif q == 5:
    rangeLimit = 4
elif q == 4:
    rangeLimit = 5
else:
    rangeLimit = 10

# evaluate evenly spaced points in k-space.
# note the offset to avoid the singularity
# and keep the momentum points in a separate file
for i in range(rangeLimit):
    for r in range(rangeLimit):
        momentum = 2*np.pi*i/rangeLimit * tb1 + 2*np.pi*r/rangeLimit * tb2
        eigvals = eigvalsh(generateHamiltonian(momentum + np.array([0.0002,0.0002])))
        histPoints+=list(eigvals)
        kPoints += [momentum for i in range(len(eigvals))]

# save the file.
np.save('rawData/data_'+str(p)+'_'+str(q)+'.npy',histPoints)
np.save('rawData/points_'+str(p)+'_'+str(q)+'.npy',kPoints)
