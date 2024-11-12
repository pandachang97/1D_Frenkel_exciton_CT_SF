import json
import numpy as np
import time as tm
from Hamiltonian import *
from basis_set import IBS
import time as tm
###Generating and diagonalize Hamilitonian

# Open the file and load the data as a dictionary

with open('parameters.json') as DHf:
    parameters = json.load(DHf)

    check_eigenvector = parameters["printing_options"]['check_eigenvector']
    debug = parameters["printing_options"]['debug']
    eV = parameters["Energy_unit_exchange"]['eV']
    vib_freq = parameters["Energy_unit_exchange"]['vib_freq']

class DEE:
    def __init__( self ):

        self.check_eigenvector = check_eigenvector
        self.debug = debug

    def Diag_Ham(self):
        ##step 2, generating Hamiltonian and diagonalization
        #step 2.1, check the diabatic Hamiltonian
        start_time = tm.time()
        my_Ham = GHam()


        self.diabatic_Ham , self.kcount = my_Ham.gen_Ham()
        print (f'The total dimension of Hamiltonian in Diag_Ham is', self.kcount)
        end_time = tm.time()
        ##Checking if the Hamiltonian is symmetric
        tol = 1e-12
        if (  np.all( ( self.diabatic_Ham - self.diabatic_Ham.T) < tol)  ):
            print(f'The Hamiltonian matrix is symmetric')
            print(f'Time consumed by generating Hamiltonian is', end_time - start_time )
        else:
            print("Stupid Hairy Monkey, check your Hamiltonian")
        if not np.allclose(self.diabatic_Ham, self.diabatic_Ham.T):
            raise ValueError("The Hamiltonian matrix is not symmetric, which may result in complex eigenvalues and eigenvectors.")
        else:
            print(f'The Hamiltonian matrix is symmetric')
##Check the Hamiltonian if all elements are right
        if self.debug:
            np.savetxt('dia_Ham_output.txt', self.diabatic_Ham, fmt='%2.6f', delimiter=' ')
            print("Array saved to dia_Ham_output.txt")
    # step 2.2, diagonalization of the adiabatic Hamiltonian
    ##initialize the eigenvalue and eigenvector matrix
        self.evect = np.zeros( (self.kcount, self.kcount) , dtype=float )
        self.evalue = np.zeros( ( self.kcount ), dtype=float )
        start_time = tm.time()
        #self.evalue = np.zeros( ( self.kcount ), dtype=float )
        #self.evect = np.zeros( (self.kcount , self.kcount), dtype=float )
        ##diagonalization
        self.evalue.real, self.evect.real = np.linalg.eig(self.diabatic_Ham)
        #self.evalue, self.evect = np.linalg.eigh(self.diabatic_Ham)
        end_time = tm.time()
        print(f'Time consumed by diagonalizing Hamiltonian is', end_time - start_time )
        # step 2.3, change energy unit into electronic voltage(eV) 
        for i in range( 0 , self.kcount):
            self.evalue[ i ] = self.evalue[ i ] * np.divide(  vib_freq , eV , dtype=float )

        print(f'The eigenvalue has been changed to eV here!!!')

        if (self.check_eigenvector):
            np.savetxt('eigenvalue_output.txt', self.evalue, fmt='%4.6f', delimiter=' ')
            print("Array saved to eigenvalue_output.txt")
            np.savetxt('eigenvector_output.txt', self.evect, fmt='%2.6f', delimiter=' ')
            print("Array saved to igenvector_output.txt")
#evalue = np.real_if_close(evalue)
#evect = np.real_if_close(evect, tol= 1e-01)
        return self.evalue , self.evect , self.kcount

