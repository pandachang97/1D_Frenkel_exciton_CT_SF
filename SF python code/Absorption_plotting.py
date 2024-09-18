##Calculating absorption spectrum
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
step = parameters["Abs_plotting"]['step']
step_width = parameters["Abs_plotting"]['step_width']
le_sigma = parameters["Abs_plotting"]['le_sigma']
he_sigma = parameters["Abs_plotting"]['he_sigma']
wcut = parameters["Abs_plotting"]['wcut']
Nomalized = parameters["Abs_plotting"]['Nomalized']
E_ex_s = parameters["Energy_setting"]['E_ex_s'] #Excat exciton energy in TDDFT or MBPT

# Define HR_fac class
class Abs:
    def __init__(self, kcount , evalue , OSC_X, OSC_Y ):
        self.LamGE_S = LamGE_S
        self.Nchrom = Nchrom
        self.vibmax = vibmax
        self.kcount = kcount
        self.evalue = evalue
        self.step = step
        self.step_width = step_width
        self.Nomalized = Nomalized
        self.ABS_X = np.zeros( self.step , dtype=float )
        self.ABS_Y = np.zeros( self.step , dtype=float )
        self.ABS_TOTAL = np.zeros( self.step , dtype=float )
        self.E_ex_s = E_ex_s
        self.x = np.zeros(self.step)
        self.wcut = wcut
        self.le_sigma = le_sigma
        self.he_sigma = he_sigma
        self.OSC_X = OSC_X
        self.OSC_Y = OSC_Y
        self.mu = 0.e0
        

    def cal_ABS(self):
        for i in range( 0 , self.step):
            self.x[i] = self.E_ex_s + np.multiply(self.step_width, i - self.step/2 , dtype=float)
            for j in range( 0 , self.kcount):
                self.mu = self.E_ex_s + self.evalue[j]
                if (self.mu < self.wcut): ##practical plotting
                    self.ABS_X[i] = self.ABS_X[i] + self.OSC_X[j] * np.exp(-( ( self.x[i] - self.mu ) / self.le_sigma )**2 )
                    #print(f'OSC_X on X is ', self.OSC_X[j]) 
                    #print(f'absortion on X is ', self.ABS_X[i])  
                    self.ABS_Y[i] = self.ABS_Y[i] + self.OSC_Y[j] * np.exp(-( ( self.x[i] - self.mu ) / self.le_sigma )**2 )  
                else:
                    self.ABS_X[i] = self.ABS_X[i] + self.OSC_X[j] * np.exp(-( ( self.x[i] - self.mu ) / self.he_sigma )**2 )  
                    self.ABS_Y[i] = self.ABS_Y[i] + self.OSC_Y[j] * np.exp(-( ( self.x[i] - self.mu ) / self.he_sigma )**2 )  
            for i in range( 0 , self.step):
                self.ABS_TOTAL[i] = self.ABS_X[i] + self.ABS_Y[i]
                
        print(f'The absortion is done')
        print(f'The dimenstion of ABS_TOTAL is' , np.shape(self.ABS_TOTAL) )
        return self.x, self.ABS_TOTAL 
        