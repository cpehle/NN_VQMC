import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import time

# Define Pauli operators
Id = sps.eye(2)
Sx = sps.csr_matrix(np.array([[0,1.],[1,0]]))
Sz = sps.csr_matrix(np.array([[1,0.],[0,-1]]))
Sy = -1j * Sz.dot(Sx)


#--------------------------------------------------------------
# Indexing function for Boundary conditions 

def index(x,y,L):
    if (x<0):
        x += L
    if (x>=L):
        x -= L
    if (y<0):
        y += L
    if (y>=L):
        y -= L
 
    return L*y+x

#--------------------------------------------------------------
# Build up the Square lattice 

def buildSquareLattice(L):
    
    N = L*L
    Neighbors   = np.zeros((N,4))
    Coordinates = np.zeros((N,2))
    counter = np.zeros(2)
    
    for i in range(1,N):
        
        if ((counter[0]+1) % L == 0):

            counter[0]=0

            if ((counter[1]+1) % L == 0 ):
                counter[1] = 0
            else:
                counter[1]+=1

        else:
            counter[0]+=1

        Coordinates[i,0] = counter[0]
        Coordinates[i,1] = counter[1]

    for i in range(N):

        Neighbors[i,0] = index(Coordinates[i,0]+1,Coordinates[i,1],L)  
        Neighbors[i,1] = index(Coordinates[i,0]  ,Coordinates[i,1]+1,L)  
        Neighbors[i,2] = index(Coordinates[i,0]-1,Coordinates[i,1],L)  
        Neighbors[i,3] = index(Coordinates[i,0]  ,Coordinates[i,1]-1,L)  

    return Neighbors

#--------------------------------------------------------------
# Build up interaction term between site i and external field h

def buildMagneticX(i,h,N):
    OpList = []
    for k in range(N):
    	if (k == i):
    		OpList.append(Sx)
    	else:
    		OpList.append(Id)
    
    return h*reduce(sps.kron,OpList)


#--------------------------------------------------------------
# Build up interaction between site i and j

def buildInteractionZZ(i,j,J,N):
	
    OpList = []
    for k in range(N):
    
        if (k == i):
            OpList.append(Sz)
        elif (k == j):
            OpList.append(Sz)
        else:
            OpList.append(Id)
    
    return J*reduce(sps.kron,OpList)


#--------------------------------------------------------------
# Build transverse-field Ising model

def build2dIsingModel(L,J,h):

    N = L*L
    Dim = 2**N
     
    neigh = buildSquareLattice(L)
    
    Ham = sps.csr_matrix(np.zeros((Dim,Dim)))
    for i in range(N):
        for j in range(2):
    	    Ham = Ham - buildInteractionZZ(i,neigh[i,j],J,N)
    	    Ham = Ham - buildMagneticX(i,h,N)	
    
    return Ham


#--------------------------------------------------------------
# Main function

def main():
	


    # Hamiltonian parameters
    #h = 1.0
    J = 1.0
    L = 2
    N = L*L 
    
    fout = open("../data/2dTFIM_square/ed-dmrg/Energy_ed_2dTFIM_Square_L2.dat",'w')


    for i in range(21):
        h = 0.0 + 0.1*i
     

        Hamiltonian = build2dIsingModel(L,J,h)
     
        HamDense = np.asarray(Hamiltonian.todense())
        e = np.linalg.eigvalsh(HamDense)
        fout.write('%f     %f\n' % (h,e[0]/N))
        #print e[0]/N

    



if __name__ == "__main__":

	main()

