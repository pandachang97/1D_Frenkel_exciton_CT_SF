##Calculating emission spectrum
import math
import json
import numpy as np
import time as tm
from Emission_OSC import EMI_OSC

with open('parameters.json') as EMI_inp:
    parameters = json.load(EMI_inp)

    Nchrom = parameters['geometry_parameters']['Nchrom']
    vibmax = parameters['geometry_parameters']['vibmax']
    theta = parameters['geometry_parameters']['theta']

    LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']

    step = parameters["Abs_plotting"]['step']
    step_width = parameters["Abs_plotting"]['step_width']

    Nomalized = parameters["Abs_plotting"]['Nomalized']
    E_ex_s = parameters["Energy_setting"]['E_ex_s'] #Excat exciton energy in TDDFT or MBPT
    emission_gamma = parameters['Emi_plotting']['emission_gamma']
    emi_freq_fac_switch = parameters['Emi_plotting']['emi_freq_fac_switch']
    Emi_Temp_depen = parameters['Emi_plotting']['Emi_Temp_depen']
    Initial_Temp = parameters['Emi_plotting']['Initial_Temp']
    Temp_step = parameters['Emi_plotting']['Temp_step']
    TOT_N_Temp = parameters['Emi_plotting']['TOT_N_Temp']
    Kb = parameters['Emi_plotting']['Kb']
    

    eV = parameters["Energy_unit_exchange"]['eV']
    vib_freq = parameters["Energy_unit_exchange"]['vib_freq']


class EMI_SP:
    def __init__(self, kcount , evect, evalue  ):
        self.LamGE_S = LamGE_S
        self.Nchrom = Nchrom
        self.vibmax = vibmax
        self.kcount = kcount
        self.evect = evect
        self.evalue = evalue
        self.step = step
        self.step_width = step_width
        self.Nomalized = Nomalized
        self.EMI_X = np.zeros( self.step , dtype=float )
        self.EMI_Y = np.zeros( self.step , dtype=float )
        #self.EMI_TOTAL = np.zeros( self.step , dtype=float )
        self.E_ex_s = E_ex_s
        self.Emi_Temp_depen = Emi_Temp_depen
        self.x = np.zeros(self.step)
        self.Initial_Temp = Initial_Temp
        self.Temp_step = Temp_step
        self.TOT_N_Temp = TOT_N_Temp
        self.Kb = Kb
        self.emi_freq_fac_switch = emi_freq_fac_switch
        self.mu = 0.e0
        self.vib_freq = vib_freq
        self.eV = eV
        self.emission_gamma = emission_gamma
        self.fun_Z = np.zeros( TOT_N_Temp, dtype=float )
        self.Bfac = np.zeros(self.kcount , dtype=float)
        self.fd = 0.e0

    ## Bolzmann factor 
    def BZ_fuc(self, k , tkelvin ):
        self.fd = np.exp( -1.e0 * ( self.evalue[k] - np.min( self.evalue ) ) /( self.Kb * tkelvin )   )
        return self.fd
    
    #!! Bolzmann Ensemble function 
    def Bol_Z(self):
        tkelvin = 0.e0
        if (self.Emi_Temp_depen):
            Number_of_Temp = self.TOT_N_Temp
        else:
            Number_of_Temp = 1
        for ktem in range( 0 , Number_of_Temp ):  
            tkelvin = np.multiply( ktem , self.Temp_step , dtype=float ) + self.Initial_Temp
            for i in range( 0 , self.kcount ):
                if ( tkelvin == 0.0e00):
                    self.fun_Z[ ktem ] = 1.0e0
                else:
                    self.fun_Z[ ktem ] = self.fun_Z[ ktem ] + self.BZ_fuc( i , tkelvin )
        return self.fun_Z 


    def cal_EMI(self):
        self.Dis_Z = self.Bol_Z()
        start_time = tm.time()
        my_EMI = EMI_OSC ( self.kcount , self.evect , self.evalue )
        Emi_osc_x , Emi_osc_y = my_EMI.gen_EMI_OSC()
        end_time = tm.time()
        print(f'Time of calculting the oscillator strength of Emission spectrum is ', end_time - start_time )
        start_time = tm.time()
        if ( self.Emi_Temp_depen and self.TOT_N_Temp > 1 ):
            self.EMI_TOTAL = np.zeros( ( self.step , self.TOT_N_Temp ), dtype=float )
            for N_Temp in range( 0 , self.TOT_N_Temp):
                self.EMI_X = np.zeros( self.step , dtype=float )
                self.EMI_Y = np.zeros( self.step , dtype=float )
                tkelvin = np.multiply( N_Temp , self.Temp_step , dtype=float ) + self.Initial_Temp
                for i in range( 0 , self.step):
                    self.x[i] = self.E_ex_s + np.multiply(self.step_width, i - self.step/2 , dtype=float)
                    for j in range( 0 , self.kcount):
                        if ( tkelvin == 0.0e0):
                            self.Bfac[j] = 1.0e0
                        else:
                            self.Bfac[j] = self.BZ_fuc( j , tkelvin ) / self.Dis_Z[N_Temp]
                            self.mu = ( self.E_ex_s + self.evalue[j]  ) 
                        for j1 in range( 0 , vibmax + 1 ):
                            if ( self.emi_freq_fac_switch ): ##practical plotting
                                emi_freq_fac = (self.E_ex_s + self.evalue[ j ] - j1 * self.vib_freq / self.eV ) **3
                            else:
                                emi_freq_fac = 1.0e0
                            

                            self.EMI_X[i] = self.EMI_X[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_x[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )
  
                            self.EMI_Y[i] = self.EMI_Y[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_y[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )  


                self.EMI_TOTAL[:,N_Temp] = self.EMI_X + self.EMI_Y
            print(f'The dimenstion of ABS_TOTAL is' , np.shape(self.EMI_TOTAL) )
        else:
                self.EMI_TOTAL = np.zeros(self.step , dtype=float )
                self.EMI_X = np.zeros( self.step , dtype=float )
                self.EMI_Y = np.zeros( self.step , dtype=float )
                tkelvin = self.Initial_Temp
                for i in range( 0 , self.step):
                    self.x[i] = self.E_ex_s + np.multiply(self.step_width, i - self.step/2 , dtype=float)
                    for j in range( 0 , self.kcount):
                        if ( tkelvin == 0.0e0):
                            self.Bfac[j] = 1.0e0
                        else:
                            self.Bfac[j] = self.BZ_fuc( j , tkelvin ) / self.Dis_Z[0]
                            self.mu = ( self.E_ex_s + self.evalue[j]  ) 
                        for j1 in range( 0 , vibmax + 1 ):
                            if ( self.emi_freq_fac_switch ): ##practical plotting
                                emi_freq_fac = (self.E_ex_s + self.evalue[ j ] - j1 * self.vib_freq / self.eV ) **3
                            else:
                                emi_freq_fac = 1.0e0
                            

                            self.EMI_X[i] = self.EMI_X[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_x[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )
  
                            self.EMI_Y[i] = self.EMI_Y[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_y[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )  

                self.EMI_TOTAL = self.EMI_X + self.EMI_Y
        end_time = tm.time()        
        print(f'The emission spectrum is done and the time is', end_time - start_time )
        #print(f'The dimenstion of ABS_TOTAL is' , np.shape(self.EMI_TOTAL) )
        return self.EMI_TOTAL 
        