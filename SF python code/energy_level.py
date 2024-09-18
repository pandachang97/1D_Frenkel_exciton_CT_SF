import json

with open('parameters.json') as EL_f:
    parameters = json.load(EL_f)

Del_LUMO = parameters["Energy_setting"]['Del_LUMO']
Del_HOMO = parameters["Energy_setting"]['Del_HOMO']
ESex_ref = parameters["Energy_setting"]['ESex_ref'] # singlet exciton energy in code
K_exchange = parameters["Energy_setting"]['K_exchange']  #exchange energy, ETex_ref = ESex_ref + K_exchange (K<0 generally)
VMU_inter=parameters["Energy_setting"]['VMU_inter']

class EL:
    def __init__( self ):
        print(ESex_ref, Del_LUMO, Del_HOMO, K_exchange, VMU_inter)
        self.ESex_ref = ESex_ref
        self.Del_LUMO = Del_LUMO
        self.Del_HOMO = Del_HOMO
        self.K_exchange = K_exchange
        self.VMU_inter = VMU_inter


    def calcEPpPn(self):
        return self.ESex_ref - self.Del_LUMO + self.VMU_inter

    def calcEPnPp(self):
        return self.ESex_ref + self.Del_HOMO + self.VMU_inter

    def calcEPtPt(self):
        return self.ESex_ref + self.K_exchange

