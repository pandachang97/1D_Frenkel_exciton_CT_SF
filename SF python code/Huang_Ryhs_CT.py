##Calculating HR factor for CT to Ground and exciton to CT
import math
import json

# Read parameters from JSON file
with open('parameters.json') as HRf:
    parameters = json.load(HRf)
LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']
LamGE_T = parameters["huang_ryhs_factors"]['LamGE_T']

# Define HR_fac class
class HR_fac:
    def __init__(self):
        self.LamGE_S = LamGE_S
        self.LamGE_T = LamGE_T
    
    def calculate_LamCTG(self):
        return math.sqrt(0.5) * self.LamGE_S

    def calculate_LamECT(self):
        return self.LamGE_S -math.sqrt(0.5) * self.LamGE_S

    def calculate_LamCTpT(self): ##deviation of + state of CT to Triplet pair state
        return math.sqrt(0.5) * self.LamGE_T
    
    def calculate_LamCTnT(self): ##deviation of - state of CT to Triplet pair state
        return self.LamGE_T -math.sqrt(0.5) * self.LamGE_T