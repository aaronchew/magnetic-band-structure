import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, loggamma
from scipy.linalg import eigvalsh
from mpmath import jtheta,sqrt
import csv
from scipy.sparse.linalg import expm
import sys
from math import gcd

coolio = sys.argv
p=int(coolio[1])
q=int(coolio[2])
phi = (2*np.pi*p)/q
w = 7.0 # the potential strength

print(p, q)

# calculates the Siegel theta function from Jacobis
def siegel(z1,z2):
    return(np.sqrt(2)*np.exp(-np.pi*z1*z1-np.pi*z2*z2+2j*np.pi*z1*z2)* \
    jtheta(3,np.pi*(1.0j*z1+z2),np.exp(-np.pi))*jtheta(3,np.pi*(z1+1.0j*z2),np.exp(-np.pi)))

# get the phase e^{\xi_q(k)}
def getPhase(kvec,qvec,latticeVec1,latticeVec2):
    k1 = np.dot(kvec,latticeVec1)
    k2 = np.dot(kvec,latticeVec2)
    q1 = np.dot(qvec,latticeVec1)
    q2 = np.dot(qvec,latticeVec2)
    
    qComplex = q1 + 1.0j*q2
    numer = np.exp(-qComplex*np.conj(qComplex)/(8*np.pi))*siegel((k1-qComplex/2)/(2*np.pi),(k2+1.0j*qComplex/2)/(2*np.pi))
    denom= sqrt(siegel(k1/(2*np.pi),k2/(2*np.pi))*siegel((k1-q1)/(2*np.pi),(k2-q2)/(2*np.pi)))
    return(complex(numer/denom))

a1 = np.array([1,0],dtype=complex)
a2 = np.array([0,1],dtype=complex)
b1 = np.array([1,0],dtype=complex)
b2 = np.array([0,1],dtype=complex)

ta1 = 1.0/p*a1
ta2 = q*a2
tb1 = p*b1
tb2 = 1.0/q*b2

omega = ta1[0]*ta2[1] - ta1[1]*ta2[0]
z1 = 1/np.sqrt(omega)*(ta1[0]+1.0j*ta1[1])
z2 = 1/np.sqrt(omega)*(ta2[0]+1.0j*ta2[1])

qvec = 2*np.pi*b1
gQ = np.dot(qvec,ta1)*np.conj(z2)-np.dot(qvec,ta2)*np.conj(z1)
gQbar = np.conj(gQ)

num_landau_calc = 200
num_landau = 100

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
        
kinetic = np.diag(np.array([phi*(m + 0.5) for m in range(num_landau)]))

def generateHamiltonian(kvec):
    diagKinetic = np.zeros((p*num_landau,p*num_landau),dtype=complex)
    diagPotential = np.zeros((p*num_landau,p*num_landau),dtype=complex)
    offDiagPotential = np.zeros((p*num_landau,p*num_landau),dtype=complex)

    for v in range(p):
        diagKinetic[v*num_landau:(v+1)*num_landau,v*num_landau:(v+1)*num_landau] = kinetic
        diagPotential[v*num_landau:(v+1)*num_landau,v*num_landau:(v+1)*num_landau] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b2,ta1,ta2)*hqp2 
        
        # -1 entry is last entry
        if v == 0:
            offDiagPotential[(v-1)*num_landau:,v*num_landau:(v+1)*num_landau] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b1,ta1,ta2)*hqp1
        else:
            offDiagPotential[(v-1)*num_landau:(v)*num_landau,v*num_landau:(v+1)*num_landau] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b1,ta1,ta2)*hqp1

    potentialTerm = diagPotential + offDiagPotential
    hamiltonian = diagKinetic + w/2*(potentialTerm + potentialTerm.conj().T)    
    return(hamiltonian)


histPoints = []
for i in range(10):
    for r in range(10):
        momentum = 2*np.pi*i/10 * tb1 + 2*np.pi*r/10 * tb2
        eigvals = eigvalsh(generateHamiltonian(momentum))
        histPoints+=list(eigvals)
        
np.save('rawData/data_'+str(p)+'_'+str(q)+'.npy',histPoints)