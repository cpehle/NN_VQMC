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
# Build up interaction term between site i and external field h

def buildMagneticX(i,h,L):
	OpList = []
	for k in range(L):
		if (k == i):
			OpList.append(Sx)
		else:
			OpList.append(Id)

	return h*reduce(sps.kron,OpList)


#--------------------------------------------------------------
# Build up interaction between site i and j

def buildInteractionZZ(i,j,J,L):
	OpList = []
	for k in range(L):
		if (k == i):
			OpList.append(Sz)
		elif (k == j):
			OpList.append(Sz)
		else:
			OpList.append(Id)

	return J*reduce(sps.kron,OpList)


#--------------------------------------------------------------
# Build transverse-field Ising model

def build1dIsingModel(L,J,h):

	Dim = 2**L

	Ham = sps.csr_matrix(np.zeros((Dim,Dim)))
	
	for i in range(L-1):
		Ham = Ham - buildInteractionZZ(i,i+1,J,L)
		Ham = Ham - buildMagneticX(i,h,L)	
	Ham = Ham - buildMagneticX(L-1,h,L)
	
	# Periodic Boundary Conditions
	#Ham = Ham - buildInteractionZZ(0,L-1,J,L)
	return Ham


#--------------------------------------------------------------
# Main function

def main():
			
	# Hamiltonian parameters
	h = 1.0
	J = 1.0
	
	# Maximum order in the cluster expansion
	MaxOrder = 14
	
	file = open("energy_ed_1d_TFIM.dat", "w")
	file.write('L\t\tE\n')

        
        L = 4;
        Hamiltonian = build1dIsingModel(L,J,h)
        HamDense = np.asarray(Hamiltonian.todense())
        e = np.linalg.eigvalsh(HamDense)
        #(e,psi) = spslin.eigsh(Hamiltonian, k=L-1, which='SA', return_eigenvectors = False)
        print e[0]/L
        #print 'Ground State Energy = %f\n' % e[0]/L

	#for i in range(1,MaxOrder+1):
	#	
	#	start_time = time.time()
	#	# Spin chain length
	#	L = i
	#	print 'Lattice Size = %i\n' % L
	#	Hamiltonian = build1dIsingModel(L,J,h)
	#	#print Ham.todense()
	#	
	#	# Diagonalize the sparse Hamiltonian
	#	#(e,psi) = spslin.eigsh(Hamiltonian, k=i-1, which='SA', return_eigenvectors = False)

	#	# Dense conversion suitable for small sizes
	#	HamDense = np.asarray(Hamiltonian.todense())
	#	#(e,psi) = np.linalg.eigh(HamDense)
	#	
	#	e = np.linalg.eigvalsh(HamDense)
	#	#print e[0]
	#	
	#	print 'Ground State Energy = %f\n' % e[0]
	#	#print 'Elapsed time: %s seconds\n' % (time.time() - start_time) 
	#	file.write('%i\t' % L)
	#	file.write('%f\n' % e[0])
	#
	#file.close()

if __name__ == "__main__":

	#initial_time = time.time()
	main()
	#print("Total time to solution: %s seconds" % (time.time() - initial_time))

