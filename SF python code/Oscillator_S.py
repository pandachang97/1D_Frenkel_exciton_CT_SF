##Calculating oscillator strength of single exciton
import math
import json
import numpy as np
import scipy
from basis_set import IBS
from FC_factor import FCF


with open('parameters.json') as OSC_inp:
    parameters = json.load(OSC_inp)

Nchrom = parameters['geometry_parameters']['Nchrom']
vibmax = parameters['geometry_parameters']['vibmax']
theta = parameters['geometry_parameters']['theta']

LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']
abs_freq_fac_switch = parameters["Abs_plotting"]['abs_freq_fac_switch']
CT_inter_periodic = parameters["basis_set_options"]['CT_inter_periodic']
E_ex_s = parameters["Energy_setting"]['E_ex_s'] #Excat exciton energy in TDDFT or MBPT

# Define HR_fac class
class OSC:
    def __init__(self, kcount,  evect, evalue ):
        self.LamGE_S = LamGE_S
        self.Nchrom = Nchrom
        self.vibmax = vibmax
        self.kcount = kcount
        self.evect = evect
        self.evalue = evalue
        self.theta = theta
        
        self.abs_freq_fac_switch = abs_freq_fac_switch
        self.TDM_X = np.zeros( self.kcount , dtype=float )
        self.TDM_Y = np.zeros( self.kcount , dtype=float )
        self.OSC_X = np.zeros( self.kcount , dtype=float )
        self.OSC_Y = np.zeros( self.kcount , dtype=float )
        self.CT_inter_periodic = CT_inter_periodic

    def cal_OSC(self):
        OSC_IBS = IBS()
        self.Index_single = OSC_IBS.arr_1p()
        OSC_FCF = FCF()
        self.theta = np.divide(self.theta, 180 , dtype=float) * np.pi  ##change the angle into radius unit
        for i in range( 0 , self.kcount ):
            for l in range( 0 , self.Nchrom):
                for l1 in range( 0 , self.vibmax + 1 ):
                    num_1p = OSC_IBS.order_1p( l , l1 )
                    lab2 = self.Index_single[num_1p]  
                    #print(f'the Index_single element in OSC_abs is', self.Index_single[num_1p] )
                    self.TDM_X[i] = self.TDM_X[i] + math.cos( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
                    #print(f'the eigenvector is', self.evect[lab2 , i] )
                    #print(f'the FC_factor is', OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S ) )
                    #print(f'the TDM_X is', self.TDM_X[i] )
                    self.TDM_Y[i] = self.TDM_Y[i] + math.sin( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
        
        ##Now, we are calculating oscillator strength
        for i in range( 0 , self.kcount):
            if (self.abs_freq_fac_switch): ##practical plotting
                self.OSC_X[i] = np.multiply ( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_X[i]**2 ) , dtype=float )
                #print(f'the OSC_X is', self.OSC_X [ i ] )
                self.OSC_Y[i] = np.multiply( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_Y[i]**2 ) , dtype=float )
            else:
                self.OSC_X[i] = self.OSC_X[i] **2  
                self.OSC_Y[i] = self.OSC_Y[i] **2  
                
        print(f'The oscillater strength is done')
        return self.OSC_X , self.OSC_Y
        
    def cal_OSC_se(self):

        OSC_IBS_se = IBS(self.Nchrom , self.vibmax)
        OSC_FCF = FCF()
        
        Index_double = OSC_IBS_se.arr_2p()
        Index_tripple = OSC_IBS_se.arr_3p()
        for i in range(0 , 740):
            for l in range( 0 , self.Nchrom):
                for l1 in range( 0 , self.vibmax + 1 ):
                    num_1p = OSC_IBS_se.order_1p( l , l1 )
                    lab2 = self.Index_single[num_1p]  
                    self.TDM_X[i] = self.TDM_X[i] + math.cos( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
                    #print(f'the eigenvector is', self.evect[lab2 , i] )
                    #print(f'the FC_factor is', OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S ) )
                    #print(f'the TDM_X is', self.TDM_X[i] )
                    self.TDM_Y[i] = self.TDM_Y[i] + math.sin( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )

        for i in range( 0 , self.kcount):
            if (self.abs_freq_fac_switch): ##practical plotting
                self.OSC_X[i] = np.multiply ( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_X[i]**2 ) , dtype=float )
                #print(f'the OSC_X is', self.OSC_X [ i ] )
                self.OSC_Y[i] = np.multiply( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_Y[i]**2 ) , dtype=float )
            else:
                self.OSC_X[i] = self.OSC_X[i] **2  
                self.OSC_Y[i] = self.OSC_Y[i] **2  
                
        print(f'The SE oscillater strength is done')
        return self.OSC_X , self.OSC_Y

    def cal_OSC_CT(self):
        OSC_IBS_CT = IBS(self.Nchrom , self.vibmax)
        OSC_FCF = FCF()
        Index_CT = OSC_IBS_CT.arr_CT()
        Index_CTv = OSC_IBS_CT.arr_CTv()
        for i in range(740 , 1320):
            for l in range( 0 , self.Nchrom):
                for l1 in range( 0 , self.vibmax + 1 ):
                    num_1p = OSC_IBS_CT.order_1p( l , l1 )
                    lab2 = self.Index_single[num_1p]  
                    self.TDM_X[i] = self.TDM_X[i] + math.cos( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
                    #print(f'the eigenvector is', self.evect[lab2 , i] )
                    #print(f'the FC_factor is', OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S ) )
                    #print(f'the TDM_X is', self.TDM_X[i] )
                    self.TDM_Y[i] = self.TDM_Y[i] + math.sin( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )

        for i in range( 0 , self.kcount):
            if (self.abs_freq_fac_switch): ##practical plotting
                self.OSC_X[i] = np.multiply ( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_X[i]**2 ) , dtype=float )
                #print(f'the OSC_X is', self.OSC_X [ i ] )
                self.OSC_Y[i] = np.multiply( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_Y[i]**2 ) , dtype=float )
            else:
                self.OSC_X[i] = self.OSC_X[i] **2  
                self.OSC_Y[i] = self.OSC_Y[i] **2  
        print(f'The CT oscillater strength is done')
        return self.OSC_X , self.OSC_Y


    def cal_OSC_TP(self):
        OSC_IBS_TP = IBS(self.Nchrom , self.vibmax)
        OSC_FCF = FCF()
        #Index_TP = OSC_IBS_TP.arr_TP()
        #Index_TPv = OSC_IBS_TP.arr_TPv()

     #   for i in range(0 , self.Nchrom ):
     #       for i1 in range(0 , self.vibmax + 1):
     #           for j in range( 0 , self.Nchrom ):
     #               for j1 in range( 0 , self.vibmax + 1 ):
     #                   if i >= j:
     #                       continue
     #                   else:
     #                       if( ( ( abs( i - j ) == 1 ) ) or  \
     #                       ( self.Nchrom >= 3 and abs( i - j ) == self.Nchrom - 1 and self.CT_inter_periodic ) ):
     #                           if ( i1 + j1 > self.vibmax ):
     #                               continue
     #                           else:
     #                               num_CT = OSC_IBS_TP.order_CT(i , i1, j , j1)
     #                               lab1 = OSC_IBS_TP.Index_TP[num_CT]
     #                               for l in range( 0 , self.Nchrom):
     #                                   for l1 in range( 0 , self.vibmax + 1 ):
    #                                        num_1p = self.order_1p( l , l1 )
     #                                       lab2 = self.Index_single[num_1p]  
     #                                       self.TDM_X[i] = self.TDM_X[i] + math.cos( l * self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
        for i in range(1320 , self.kcount):
            for l in range( 0 , self.Nchrom):
                for l1 in range( 0 , self.vibmax + 1 ):
                    num_1p = OSC_IBS_TP.order_1p( l , l1 )
                    lab2 = self.Index_single[num_1p]  
                    self.TDM_X[i] = self.TDM_X[i] + math.cos( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )
                    #print(f'the eigenvector is', self.evect[lab2 , i] )
                    #print(f'the FC_factor is', OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S ) )
                    #print(f'the TDM_X is', self.TDM_X[i] )
                    self.TDM_Y[i] = self.TDM_Y[i] + math.sin( self.theta) * self.evect[lab2 , i] * OSC_FCF.gen_FCF( 0 , l1 , self.LamGE_S )

        for i in range( 0 , self.kcount):
            if (self.abs_freq_fac_switch): ##practical plotting
                self.OSC_X[i] = np.multiply ( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_X[i]**2 ) , dtype=float )
                #print(f'the OSC_X is', self.OSC_X [ i ] )
                self.OSC_Y[i] = np.multiply( ( E_ex_s + self.evalue[ i ] ) , ( self.TDM_Y[i]**2 ) , dtype=float )
            else:
                self.OSC_X[i] = self.OSC_X[i] **2  
                self.OSC_Y[i] = self.OSC_Y[i] **2  
        print(f'The CT oscillater strength is done')
        return self.OSC_X , self.OSC_Y