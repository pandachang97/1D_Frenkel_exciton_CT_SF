import numpy as np
import math
import scipy
##function to calculate Frank-Condon factor
class FCF:
    def __init__( self ):
        self.FC_fv = 0.e0

    def gen_FCF( self, m , n , lam ): 
        ## m is the vibrational quanta on lower electronic energy state
        ## n is the vibrational quanta on higher electronic energy state
        
        HR_F = np.square( lam, dtype=float ) 
        C1 = np.sqrt( math.factorial( m ) , dtype=float )
        D1 = np.sqrt( math.factorial( n ) , dtype=float )
        j = n
        S1 = 0.0
        for i in range( 1, j + 2 ):
            k = i - 1
            if ( m - n + k < 0):
                continue
            else:
                C2 = math.factorial( k )
                D2 = math.factorial ( n - k)
                C3 = math.factorial ( m - n + k )
                DEN = C2 * D2 * C3
                S1 = S1 + ((((-1)**(m-n+k))*(HR_F**((m-n+(2*k))/2.0))))
                S1 = np.divide( S1, DEN, dtype=float )
                ##print(f'S1 is' , S1 )

        self.FC_fv = C1*D1*np.exp( -1.0* (HR_F * 2.0 ) , dtype= float ) * S1
        ##print(f'FC_fv is' , self.FC_fv )
        return self.FC_fv