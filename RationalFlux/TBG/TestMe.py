import sys
import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, loggamma
from mpmath import jtheta,sqrt
import matplotlib.pyplot as plt
import csv
from math import gcd

coolio = sys.argv
p=int(coolio[1])
q=int(coolio[2])
phi = (2*np.pi*p)/q
w = 7.0 # the potential strength

assert(gcd(p,q) == 1)

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

num_landau = 100

hqp1 = np.genfromtxt('prunedCSV/formFactor1_'+str(p)+'_'+str(q)+'_mod.csv', delimiter=',',dtype='complex')
hqp2 = np.genfromtxt('prunedCSV/formFactor2_'+str(p)+'_'+str(q)+'_mod.csv', delimiter=',',dtype='complex')
        
kinetic = np.diag(np.array([phi*(m + 0.5) for m in range(num_landau)]))

def generateHamiltonian(kvec):
    diagKinetic = [[np.zeros((num_landau,num_landau)) for v in range(p)] for u in range(p)]
    diagPotential = [[np.zeros((num_landau,num_landau)) for v in range(p)] for u in range(p)]
    offDiagPotential = [[np.zeros((num_landau,num_landau)) for v in range(p)] for u in range(p)]

    for v in range(p):
        diagKinetic[v][v] = kinetic
        diagPotential[v][v] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b2,ta1,ta2)*hqp2 
        
        # -1 entry is last entry
        offDiagPotential[v-1][v] = getPhase(kvec+2*np.pi*b1*v,2*np.pi*b1,ta1,ta2)*hqp1
    
    potentialTerm = np.block(diagPotential) + np.block(offDiagPotential)
    hamiltonian = np.block(diagKinetic) + w/2*(potentialTerm + potentialTerm.conj().T)    
    return(hamiltonian)

histPoints = []
for i in range(20):
    for r in range(20):
        momentum = 2*np.pi*i/20 * tb1 + 2*np.pi*r/20 * tb2
        eigvals, eigvecs = np.linalg.eigh(generateHamiltonian(momentum))
        histPoints+=list(eigvals)
        
np.save('rawData/data_'+str(p)+'_'+str(q)+'.npy',histPoints)