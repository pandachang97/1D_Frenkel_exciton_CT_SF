import numpy as np
import math
import json
from basis_set import IBS
from FC_factor import FCF

##function to calculate emission oscillator strength

with open('parameters.json') as ESI_f:
    parameters = json.load(ESI_f)
    Nchrom = parameters['geometry_parameters']['Nchrom']
    vibmax = parameters['geometry_parameters']['vibmax']
    theta = parameters['geometry_parameters']['theta']
    LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']
    calc_Emi = parameters['Emi_plotting']['calc_Emi']
    add_double = parameters["basis_set_options"]['add_double']
    add_tripple = parameters["basis_set_options"]['add_tripple']


class EMI_OSC:
    def __init__( self, kcount , evect, evalue  ):
        self.calc_Emi = calc_Emi
        self.vibmax = vibmax
        self.Nchrom = Nchrom
        self.kcount = kcount
        self.evect = evect
        self.evalue = evalue
        self.theta =theta
        self.LamGE_S = LamGE_S
        self.add_double = add_double
        self.add_tripple = add_tripple
        self.Emi_osci_stre_x = np.zeros( ( self.kcount , self.vibmax + 1 ) , dtype=float)
        self.Emi_osci_stre_y = np.zeros( ( self.kcount , self.vibmax + 1 ) , dtype=float)

    def gen_EMI_OSC( self ): 
        EMI_IBS = IBS()
        EMI_FC = FCF()
        self.Index_single = EMI_IBS.arr_1p()
        self.Index_double = EMI_IBS.arr_2p()
        self.Index_tripple = EMI_IBS.arr_3p()
        for i in range( 0 , self.kcount):
            ##Initialize all oscillator strengths to 0
            osemy = 0.0e0
            osemx = 0.0e0
#---------- Emission to ground states with 0 vib quanta in all sites!---------------------
#      1p - 0  here, 1p is the 1p states in ground state and the exciton state
#      must be on the same site, so I just use only one set of running labels 
#-----------------------------------------------------------------------------------------
            for l in range( 0 , self.Nchrom  ):
                for l1 in range( 0 , self.vibmax + 1 ):
                    num_1p = EMI_IBS.order_1p( l , l1 )
                    lab2 = self.Index_single[num_1p]  

                    osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , l1 , self.LamGE_S )
                    osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , l1 , self.LamGE_S )

            self.Emi_osci_stre_x[ i , 0 ] = osemx **2 #I0-0
            self.Emi_osci_stre_y[ i , 0 ] = osemy **2 #I0-0
#--------- Emission to ground state with 1 site with vibrational quanta!-----------------
#  1p - 1p (with at least 1 vib in ground state) exciton 1p gruond 1p 
# Thus, they must share the same chromophore
#  2p - 1p: ground 1p exciton 1p, thus, the exciton matches a ground state
#  without vibrational quanta, the remainning pure vibrational site in exciton matches
# the 1p ground state site.   
#-----------------------------------------------------------------------------------------
#sum over ground vibrational sites 
            # 1p -1p states (at least 1 vib quanta in ground state)    
            if (self.vibmax > 0 ):
                for l in range( 0 , self.Nchrom  ):
                    for l1 in range( 0 , self.vibmax + 1 ): 
                        if ( l1 == 0 ):
                            continue
                        else:
                            osemy = 0.0e0
                            osemx = 0.0e0   
                            for m1 in range( 0 , self.vibmax + 1 ):  #vibrational quanta of exciton, ignoring the exciton position
                                num_1p = EMI_IBS.order_1p( l , m1 )
                                lab2= self.Index_single[num_1p] 
                                osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( l1 , m1 , self.LamGE_S )
                                osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( l1 , m1 , self.LamGE_S )   
    #2p - 1p, the vibrational is the same as 1p in the ground state, which are  l and l1.                             
                            if (self.add_double):
    #        !! now, I use j and j1 to index exciton position and its vibrational quanta
                                for j in range( 0 , self.Nchrom ):
                                    for j1 in range( 0 , self.vibmax + 1 ):
                                        if  l == j:
                                            continue
                                        else:
                                            if l1 + j1  > self.vibmax:
                                                continue
                                            else:  
                                                num_2p = EMI_IBS.order_2p( j , j1 , l , l1 - 1 ) 
                                                lab2 = self.Index_double[num_2p]
                                                osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S )
                                                osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S )  
                                                                                                                 
                            self.Emi_osci_stre_x[ i , l1 ] = self.Emi_osci_stre_x[ i , l1 ] + osemx **2 #I0-l1 l1 /=0 
                            self.Emi_osci_stre_y[ i , l1 ] = self.Emi_osci_stre_y[ i , l1 ] + osemy **2 #I0-l1 l1 /= 0
# Ground 2p states        
#Here, neither l1 nor m1 could be 1, otherwise, it will become 1p or no vibration state, which we have considered them in the previous case.
            if ( self.add_double ):
#        now, I use N_chain2, j and j1 to index exciton position and its vibrational quanta
                for l in range( 0 , self.Nchrom ):
                    for l1 in range( 0 , self.vibmax + 1 ):
                        for m in range( 0 , self.Nchrom ):
                            if  l >= m:
                                continue
                            else:
                                for m1 in range( 0 , self.vibmax ):
#! Cannot be on the same chromphore, but we also do not want to overcount (the second site always has to be to the right of the first)
                                    if m1 == 0: 
                                        continue
                                    else:  
                                        if ( l1 + m1 > self.vibmax ):
                                            continue
                                        else:
                                            osemy = 0.0e0
                                            osemx = 0.0e0                                              
#!Next we will not count the positon of exciton, because it will be the same as either with the first 1p ground state or with the second 1p ground state    
                                            for j1 in range( 0 , self.vibmax + 1 ):
 #!! now, we use j1 to count the vibrational quanta of exciton
 #!! cases 1, j1 stays at N_chain5 and m position with j1 vibrational quanta                                                
                                                if (j1  + l1  <= self.vibmax):
                                                    num_2p = EMI_IBS.order_2p( m , j1 , l , l1 - 1 ) 
                                                    lab2 = self.Index_double[num_2p]
                                                    osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( m1 , j1 , self.LamGE_S )
                                                    osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( m1 , j1 , self.LamGE_S )   
# !! case 2, j1 stays at N_chain4 and l position with j1 vibrational quanta
                                                if (  j1 + m1 <= self.vibmax ):
                                                    num_2p = EMI_IBS.order_2p( l , j1 , m , m1 - 1 ) 
                                                    lab2 = self.Index_double[num_2p]
                                                    osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( l1 , j1 , self.LamGE_S )
                                                    osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( l1 , j1 , self.LamGE_S ) 
                                            if ( self.add_tripple ):
#!! Here, we count the position and vibrational quanta of exciton in 3p states. 
## The other two pure vibrational states of 3p states in excitation state are the same as N_chain4 l l1 and N_chain4 m m1. 
                                                for j in range ( 0 , self.Nchrom ):
                                                    for j1 in range ( 0 , self.vibmax + 1 ):
                                                        if ( l == j or m == j ):
                                                            continue
                                                        else:
                                                            if ( j1 + m1 + l1 > self.vibmax ):
                                                                continue
                                                            else:
                                                                if ( abs( l - j ) == 1 ): 
                                                                    num_3p = EMI_IBS.order_3p( j , j1 , l, l1 - 1 , m, m1 - 1 )
                                                                    lab2 = self.Index_tripple[num_3p]
                                                                    osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S )
                                                                    osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S ) 
# !! Notice that l1 + m1 - 1 is 3 at least, that means the 2p ground state contribute to I0-2 or higher states only. 
                                            self.Emi_osci_stre_x[ i , l1 + m1  ] = self.Emi_osci_stre_x[ i , l1  + m1  ] + osemx **2 #I0-l1 + m1 - 1  l1 m1 /=0 
                                            self.Emi_osci_stre_y[ i , l1 + m1  ] = self.Emi_osci_stre_y[ i , l1  + m1  ] + osemy **2 #I0-l1 + m1 - 1  l1 m1 /= 0
#!! 3p ground state
            if ( self.add_tripple ):
                for l in range( 0 , self.Nchrom ):
                    for l1 in range( 0 , self.vibmax + 1 ):
                        if ( l1 == 0):
                             continue
                        else:
                            for m in range( 0 , self.Nchrom ):
                                if ( l >= m ):
                                     continue
                                else:
                                    for m1 in range( 0 , self.vibmax ):
                                         if ( m1 == 0 ):
                                              continue
                                         else:
                                            for n in range( 0 , self.Nchrom ):
                                                if ( m >= n):
                                                     continue
                                                else:
                                                    for n1 in range( 0 , self.vibmax ):
                                                        if ( n1 == 0):
                                                          continue
                                                        else:
                                                            if l1 + m1 + 1 + n1 > self.vibmax:
                                                                continue
                                                            else:
                                                                if ( ( abs( l - m ) == 1 ) ):
               
                                                                    osemy = 0.0e0
                                                                    osemx = 0.0e0   
#!! Here, we begin to count the vibration quanta of excited states. Since we only use the nearest site approximation, only the middle position works for the exciton
                                                                    for j1 in range( 0 , vibmax + 1 ):
                                                                        continue
                                                                    else:
 #!! exciton stays at N5, m positon with j1 vibrational quanta                                                                                       
                                                                        if ( j1  + l1  + n1  > self.vibmax ):
                                                                            num_3p = EMI_IBS.order_3p(  l, l1, m, m1, n, n1 )
                                                                            lab2 = self.Index_tripple[ num_3p ]  
                                                                            osemx = osemx + np.cos ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S )
                                                                            osemy = osemy + np.sin ( self.theta ) * self.evect[ lab2 , i ] * EMI_FC.gen_FCF( 0 , j1 , self.LamGE_S ) 

                                                                    self.Emi_osci_stre_x[ i , l1 + m1 + n1 ] = self.Emi_osci_stre_x[ i , l1 + m1 + n1 ] + osemx **2 #I0-l1 + m1 - 1 + n1 - 1  l1 m1 n1 /=0 
                                                                    self.Emi_osci_stre_y[ i , l1 + m1 + n1 ] = self.Emi_osci_stre_y[ i , l1 + m1 + n1 ] + osemy **2 #I0-l1 + m1 - 1 + n1 - 1  l1 m1 n1  /= 0
                
        return  self.Emi_osci_stre_x , self.Emi_osci_stre_y
