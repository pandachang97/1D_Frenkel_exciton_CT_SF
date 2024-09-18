import math
import numpy as np

##Define a function to calculate the Frank-Conton factor
class FC:
    def __init__( sf, vibmax, lam ):
        sf.vibmax = vibmax
        sf.lam = lam
        sf.fc_pv = np.zeros( ( sf.vibmax + 2 , sf.vibmax + 2 ) , dtype = np.float )

    def calcFC( sf, m , n ):
        m ## vib_g
        n ## vib_e
        lam = sf.lam

        S = lam*lam

        C1 = math.sqrt(math.factorial(m))
        D1 = math.sqrt(math.factorial(n))

        j = n
        Summ = 0.0
        for i in range(1, j+2):
            k = i - 1
            if (m-n+k) < 0:
                continue
            
            C2 = math.factorial(k)
            D2 = math.factorial(n-k)
            C3 = math.factorial(m-n+k)
            DEN = C2 * D2 * C3

            Summ = Summ + ((-1)**(m-n+k)) * (S**((m-n+(2*k))/2.0)) / DEN

        fc_value = C1 * D1 * math.exp(-1.0 * (S / 2.0)) * Summ
    #print('FC factor for <', m, '/', n, '> is ', FC)
    #print('lamda ', lam)
        return fc_value

##Output an array to store all possible values of FC vactor
    def store_fc( sf ):
##Let us use lower HR state index as label of row index, higher HR state index as label of colomn state.
        for i in range( 0 , sf.vibmax + 2 ):
            for j in range( 0 , sf.vibmax + 2 ):
                sf.fc_pv[ i , j ] = sf.calcFC( i , j )
        return sf.fc_pv 