##Calculating emission spectrum
import math
import json
import numpy as np
import time as tm
from Emission_OSC import EMI_OSC
from basis_set import IBS

with open('parameters.json') as EMI_inp:
    parameters = json.load(EMI_inp)

    Nchrom = parameters['geometry_parameters']['Nchrom']
    vibmax = parameters['geometry_parameters']['vibmax']
    theta = parameters['geometry_parameters']['theta']
    add_TP = parameters["basis_set_options"]['add_TP']
    add_TPv = parameters["basis_set_options"]['add_TPv']
    LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']

    step = parameters["Abs_plotting"]['step']
    step_width = parameters["Abs_plotting"]['step_width']

    Normalized = parameters["Abs_plotting"]['Normalized']
    E_ex_s = parameters["Energy_setting"]['E_ex_s'] #Excat exciton energy in TDDFT or MBPT
    emission_gamma = parameters['Emi_plotting']['emission_gamma']
    emi_freq_fac_switch = parameters['Emi_plotting']['emi_freq_fac_switch']
    Emi_Temp_depen = parameters['Emi_plotting']['Emi_Temp_depen']
    Initial_Temp = parameters['Emi_plotting']['Initial_Temp']
    Temp_step = parameters['Emi_plotting']['Temp_step']
    TOT_N_Temp = parameters['Emi_plotting']['TOT_N_Temp']
    Kb = parameters['Emi_plotting']['Kb']
    K_exchange = parameters["Energy_setting"]['K_exchange'] 

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
        self.Normalized = Normalized
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
        self.add_TP = add_TP
        self.add_TPv = add_TPv
        self.K_exchange = K_exchange
        

### If included the SF in the code, the lowest energy level will be one of the TT state. When calculating the Bolzmann factor,
### all of them will be refered to one TT state. But the exciton formed in terms of a singlet exciton, the energy is much higher 
### higher than TT state. Then, we will see nothing in the emission. Instead, I will try to find the minimum energy of singlet 
### exciton adiabatic state as the reference state. When the E-minimum is greater than zero, I will choose the Bfactor as 1.
    def search_mini_siglet(self):
        if ( self.add_TP or self.add_TPv):
            print(f'We included TT states and we will check if the TT energy level is lower than singlet exciton energy level')
            if ( self.K_exchange < 0.e0):
                EMI_IBS = IBS()
                self.Index_single = EMI_IBS.arr_1p()
                num_1p = np.count_nonzero( self.Index_single ) + 1  ### Because in python, the index of array starts from 0.
                print(f'The number sumed is ' , num_1p )
                evalue_backup = self.evalue
                while True:
                    #min_index = np.argmin(evalue_backup)
                    min_index = np.ndarray.argmin(evalue_backup)
                    den_sum =0.e0
                    for i in range( 0 , num_1p):
                        den_sum = den_sum + ( self.evect[ i , min_index ] )**2
                        
                    if ( den_sum > 4.5e-01):
                        print(f"Found a minimum value that meets the condition: {self.evalue[min_index]} at index {min_index}")
                        min_value_used = self.evalue[ min_index ]
                        break
                    else:
                        print(f"Deleted minimum value:{self.evalue[min_index]} at index {min_index}; continuing search...")  
                        evalue_backup[min_index] = 1.0e4
                    
                    
            else: 
                min_value_used = np.min( self.evalue )  
                 
        else:
            min_value_used = np.min( self.evalue )

        return min_value_used

    ## Bolzmann factor 
    def BZ_fuc(self, k , t_ke , singlet_minimum_energy ):
        
        if (  self.evalue[k] - singlet_minimum_energy >= 0.0e0 ):
            self.fd = np.exp( -1.e0 * ( self.evalue[k] - singlet_minimum_energy ) /( self.Kb * t_ke )   )
        else:
            self.fd = 1.0e0
        return self.fd
    
    #!! Bolzmann Ensemble function 
    def Bol_Z(self):
        tkelvin = 0.e0
        min_singlet_ex = self.search_mini_siglet()
        if (self.Emi_Temp_depen):
            Number_of_Temp = self.TOT_N_Temp
        else:
            Number_of_Temp = 1

        for ktem in range( 0 , Number_of_Temp ):  
            tkelvin =  ( ktem ) * self.Temp_step  + self.Initial_Temp
            print(f'The temperature is', tkelvin)
            for i in range( 0 , self.kcount ):
                if ( Number_of_Temp == 1 ):
                    self.fun_Z[ ktem ] = 1.0e0
                else:
                    self.fun_Z[ ktem ] = self.fun_Z[ ktem ] + self.BZ_fuc( i , tkelvin , min_singlet_ex )
                    ##print(f'The Bolzman Ensemble function is ', self.fun_Z)
        return self.fun_Z 


    def cal_EMI(self):
        self.Dis_Z = self.Bol_Z()
        print(f'The Bolzmann Ensemble functin is ', self.Dis_Z)
        ##print(f'The eigenvalue used in EMI_plotting is', self.evalue)
        start_time = tm.time()
        my_EMI = EMI_OSC ( self.kcount , self.evect , self.evalue )
        Emi_osc_x , Emi_osc_y = my_EMI.gen_EMI_OSC()
        end_time = tm.time()
        print(f'Time of calculting the oscillator strength of Emission spectrum is ', end_time - start_time )
        start_time = tm.time()
        min_singlet_ex1 = self.search_mini_siglet()
        if ( self.Emi_Temp_depen and self.TOT_N_Temp > 1 ):
            self.EMI_TOTAL = np.zeros( ( self.step , self.TOT_N_Temp ), dtype=float )
            for N_Temp in range( 0 , self.TOT_N_Temp):
                self.EMI_X = np.zeros( self.step , dtype=float )
                self.EMI_Y = np.zeros( self.step , dtype=float )
                tkelvin = ( N_Temp ) * self.Temp_step  + self.Initial_Temp
                
                for i in range( 0 , self.step):
                    self.x[i] = self.E_ex_s + np.multiply(self.step_width, i - self.step/2 , dtype=float)
                    for j in range( 0 , self.kcount):
                        if ( tkelvin == 0.0e00):
                            self.Bfac[j] = 1.0e0
                        else:
                            BZ_dis_num = self.BZ_fuc( j , tkelvin ,  min_singlet_ex1 )
                            ##print(f'The Bolzmann distribution for state', j ,'is', BZ_dis_num)
                            self.Bfac[j] = BZ_dis_num / self.Dis_Z[N_Temp]
                            #if( N_Temp > 0):
                               # print(f'The Bolzmann factor is', self.Bfac[j] )
                        self.mu = ( self.E_ex_s + self.evalue[j]  ) 
                        for j1 in range( 0 , vibmax + 1 ):
                            if ( self.emi_freq_fac_switch ): ##practical plotting
                                emi_freq_fac = (self.E_ex_s + self.evalue[ j ] - j1 * self.vib_freq / self.eV ) **3
                            else:
                                emi_freq_fac = 1.0e0
                            

                            self.EMI_X[i] = self.EMI_X[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_x[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )
                            
                            self.EMI_Y[i] = self.EMI_Y[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_y[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )  


                self.EMI_TOTAL[:,N_Temp] = self.EMI_X + self.EMI_Y
                data = np.column_stack((self.x, self.EMI_TOTAL))
                np.savetxt('MY_EMI.dat', data, fmt='%.6f', delimiter='\t')
            print(f'The dimenstion of ABS_TOTAL is' , np.shape(self.EMI_TOTAL) )
        else: ##single temperature 
                self.EMI_TOTAL = np.zeros(self.step , dtype=float )
                self.EMI_X = np.zeros( self.step , dtype=float )
                self.EMI_Y = np.zeros( self.step , dtype=float )
                tkelvin = self.Initial_Temp
                for i in range( 0 , self.step):
                    self.x[i] = self.E_ex_s + np.multiply(self.step_width, i - self.step/2 , dtype=float)
                    for j in range( 0 , self.kcount):
                        if ( tkelvin <= 1.0e-1):
                            self.Bfac[j] = 1.0e0
                        else:
                            self.Bfac[j] = self.BZ_fuc( j , tkelvin , min_singlet_ex1 ) / self.Dis_Z[0]

                        self.mu = ( self.E_ex_s + self.evalue[j]  ) 
                            #print(f'The Bolzman factor at', j , 'is', self.Bfac[j])
                        for j1 in range( 0 , vibmax + 1 ):
                            if ( self.emi_freq_fac_switch ): ##practical plotting
                                emi_freq_fac = (self.E_ex_s + self.evalue[ j ] - j1 * self.vib_freq / self.eV ) **3
                            else:
                                emi_freq_fac = 1.0e0
                            

                            self.EMI_X[i] = self.EMI_X[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_x[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )
                           
                            self.EMI_Y[i] = self.EMI_Y[i] + emi_freq_fac * self.Bfac[j] * Emi_osc_y[j , j1] * np.exp(-( ( self.x[i] - self.mu  + j1 * self.vib_freq / self.eV ) / self.emission_gamma )**2 )  




                self.EMI_TOTAL = self.EMI_X + self.EMI_Y

                data = np.column_stack((self.x, self.EMI_TOTAL))
                np.savetxt('MY_EMI.dat', data, fmt='%.6f', delimiter='\t')

        end_time = tm.time()        
        print(f'The emission spectrum is done and the time is', end_time - start_time )
        #print(f'The dimenstion of ABS_TOTAL is' , np.shape(self.EMI_TOTAL) )
        return self.x, self.EMI_TOTAL 
        