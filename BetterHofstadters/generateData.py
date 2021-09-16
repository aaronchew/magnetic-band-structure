import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, loggamma
from mpmath import jtheta,sqrt, pi, exp, conj
import csv
from scipy.sparse.linalg import expm
from scipy.linalg import eigvalsh
from math import gcd
import scipy as sp
import sys

coolio = sys.argv
p=int(coolio[1])
q=int(coolio[2])
phi = (2*np.pi*p)/q
w0 = 0.8/np.sqrt(3) 
w1 = 1.0/np.sqrt(3)
vfkth = 110*np.sqrt(3)

if (gcd(p,q) != 1):
    gcdVal = gcd(p,q)
    p = p//gcdVal
    q = q//gcdVal

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

a1 = 2./3*np.array([-np.sqrt(3)/2, -1./2],dtype=complex);
a2 = 2./3*np.array([np.sqrt(3)/2, -1./2],dtype=complex);
b1 = np.array([-np.sqrt(3)/2, -3./2],dtype=complex);
b2 = np.array([np.sqrt(3)/2, -3./2],dtype=complex);

ta1 = 1.0/p*a1
ta2 = q*a2
tb1 = p*b1
tb2 = 1.0/q*b2

s2 = np.array([[0, -1.0j], [1.0j, 0]],dtype=complex)
t1 = np.array([[w0, w1], [w1, w0]],dtype=complex)
t2 = np.array([[w0, w1*np.exp(-2j*np.pi/3)], [w1*np.exp(2j*np.pi/3), w0]],dtype=complex)
t3 = np.array([[w0, w1*np.exp(2j*np.pi/3)], [w1*np.exp(-2j*np.pi/3), w0]],dtype=complex)

omega = ta1[0]*ta2[1] - ta1[1]*ta2[0]
z1 = 1/np.sqrt(omega)*(ta1[0]+1.0j*ta1[1])
z2 = 1/np.sqrt(omega)*(ta2[0]+1.0j*ta2[1])

num_landau = 200

qvec = 2*np.pi*b1
gQ = np.dot(qvec,ta1)*np.conj(z2)-np.dot(qvec,ta2)*np.conj(z1)
gQbar = np.conj(gQ)

num_landau_calc = 400

toExp = np.zeros((num_landau_calc,num_landau_calc),dtype=complex)

for m in range(num_landau_calc-1):
    toExp[m,m+1] = np.sqrt(m+1)*1.0j*gQ/np.sqrt(4*np.pi)
    toExp[m+1,m] = np.sqrt(m+1)*1.0j*gQbar/np.sqrt(4*np.pi)
    
hqp1 = expm(toExp)[:num_landau,:num_landau]

qvec = 2*np.pi*b2
gQ = np.dot(qvec,ta1)*np.conj(z2)-np.dot(qvec,ta2)*np.conj(z1)
gQbar = np.conj(gQ)

toExp2 = np.zeros((num_landau_calc,num_landau_calc),dtype=complex)

for m in range(num_landau_calc-1):
    toExp2[m,m+1] = np.sqrt(m+1)*1.0j*gQ/np.sqrt(4*np.pi)
    toExp2[m+1,m] = np.sqrt(m+1)*1.0j*gQbar/np.sqrt(4*np.pi)
    
hqp2 = expm(toExp2)[:num_landau,:num_landau]
        
aUp = np.zeros((num_landau, num_landau),dtype='complex') 
aDown = np.zeros((num_landau, num_landau),dtype='complex') 

for m in range(num_landau-1):
    aUp[m+1, m] = np.sqrt(m+1)
    aDown[m,m+1] = np.sqrt(m+1)

iden = np.identity(num_landau,dtype='complex')

spurious = np.zeros((num_landau, num_landau),dtype='complex')
spurious[num_landau-1, num_landau-1] = 10

kinetic1 = np.sqrt(phi/(2*np.pi))*np.sqrt(3*np.sqrt(3)/(2*np.pi))*np.block([[spurious, aUp], [aDown, spurious]]) - 0.5*np.kron(s2,iden)
kinetic2 = np.sqrt(phi/(2*np.pi))*np.sqrt(3*np.sqrt(3)/(2*np.pi))*np.block([[spurious, aUp], [aDown, spurious]]) + 0.5*np.kron(s2,iden)

def generateHamiltonian(kvec):
    diagKinetic1 = [[np.zeros((2*num_landau,2*num_landau),dtype='complex') for v in range(p)] for u in range(p)]
    diagKinetic2 = [[np.zeros((2*num_landau,2*num_landau),dtype='complex') for v in range(p)] for u in range(p)]
    
    diagPotential = [[np.zeros((2*num_landau,2*num_landau),dtype='complex') for v in range(p)] for u in range(p)]
    offDiagPotential = [[np.zeros((2*num_landau,2*num_landau),dtype='complex') for v in range(p)] for u in range(p)]

    for v in range(p):
        diagKinetic1[v][v] = kinetic1
        diagKinetic2[v][v] = kinetic2
        
        diagPotential[v][v] = np.kron(t1,iden)+getPhase(kvec+2*np.pi*b1*v,2*np.pi*b2,ta1,ta2)*np.kron(t3,hqp2) 
        # -1 entry is last entry
        offDiagPotential[v-1][v] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b1,ta1,ta2)*np.kron(t2,hqp1)
    
    potentialTerm = np.block(diagPotential) + np.block(offDiagPotential)
    hamiltonian = np.block([[np.block(diagKinetic1),potentialTerm],[potentialTerm.conj().T,np.block(diagKinetic2)]])   
    return(vfkth*hamiltonian)

histPoints = []

# do stuff

if q == 20:
    rangeLimit = 1
elif q == 10:
    rangeLimit = 2
elif q == 5:
    rangeLimit = 4    
else:
    rangeLimit = 10

for i in range(rangeLimit):
    for r in range(rangeLimit):
        momentum = 2*np.pi*i/rangeLimit * tb1 + 2*np.pi*r/rangeLimit * tb2
        eigvals = np.linalg.eigvalsh(generateHamiltonian(momentum + np.array([0.02,0.02])))
        histPoints+=list(eigvals)
        
np.save('rawData/data_'+str(p)+'_'+str(q)+'.npy',histPoints)