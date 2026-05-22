import numpy as np
import time as tm
from basis_set import IBS
from FC_factor import FCF
from Huang_Ryhs_CT import HR_fac
from energy_level import EL
import json

with open('parameters.json') as Ham_f:
    parameters = json.load(Ham_f)

# Access the parameters as dictionary keys
    Nchrom = parameters['geometry_parameters']['Nchrom']
    vibmax = parameters['geometry_parameters']['vibmax']

    add_double = parameters["basis_set_options"]['add_double']
    add_tripple = parameters["basis_set_options"]['add_tripple']
    add_CTv = parameters["basis_set_options"]['add_CTv']
    add_CT = parameters["basis_set_options"]['add_CT']
    add_TP = parameters["basis_set_options"]['add_TP']
    add_TPv = parameters["basis_set_options"]['add_TPv']
    CT_inter_periodic = parameters["basis_set_options"]['CT_inter_periodic']
    JCou_inter_periodic = parameters["basis_set_options"]['JCou_inter_periodic']
    check_eigenvector = parameters["printing_options"]['check_eigenvector']
    debug = parameters["printing_options"]['debug']

    eV = parameters["Energy_unit_exchange"]['eV']
    vib_freq = parameters["Energy_unit_exchange"]['vib_freq']

    Del_LUMO = parameters["Energy_setting"]['Del_LUMO']
    Del_HOMO = parameters["Energy_setting"]['Del_HOMO']
    ESex_ref = parameters["Energy_setting"]['ESex_ref'] # singlet exciton energy in code
    K_exchange = parameters["Energy_setting"]['K_exchange']  #exchange energy, ETex_ref = ESex_ref + K_exchange (K<0 generally)
    JCou_inter = parameters["Energy_setting"]['JCou_inter']
    te_inter_s = parameters["Energy_setting"]['te_inter_s']
    th_inter_s = parameters["Energy_setting"]['th_inter_s']
    te_inter_t = parameters["Energy_setting"]['te_inter_t']
    th_inter_t = parameters["Energy_setting"]['th_inter_t']
    Sfactor = parameters["Energy_setting"]['Sfactor']
    VMU_inter=parameters["Energy_setting"]['VMU_inter']

    LamGE_S = parameters["huang_ryhs_factors"]['LamGE_S']
    LamGE_T = parameters["huang_ryhs_factors"]['LamGE_T']

# I want to include disorder in the future. I just define Hamiltonian and its Eigenvalue
#  as a class, then we can calculate Absorption and PL spectrum based on them.

class GHam:
    def __init__(self):
        self.Nchrom = Nchrom
        self.vibmax = vibmax
        self.kcount = 0
        self.add_double = add_double
        self.add_tripple = add_tripple
        self.add_CTv = add_CTv
        self.add_CT = add_CT
        self.add_TP = add_TP
        self.add_TPv = add_TPv
        self.eV = eV
        self.vib_freq = vib_freq
        self.CT_inter_periodic  = CT_inter_periodic
        self.JCou_inter_periodic = JCou_inter_periodic

        self.ESex_ref = ESex_ref
        self.E_CT_PpPn = 0.e0
        self.E_CT_PnPp = 0.e0
        self.E_TP = 0.e0
        self.K_exchange = K_exchange
        self.VMU_inter = VMU_inter

        self.JCou_inter = JCou_inter
        self.te_inter_t = te_inter_t
        self.th_inter_t = th_inter_t
        self.te_inter_s = te_inter_s
        self.th_inter_s = th_inter_s
        self.Sfactor = Sfactor
        self.LamGE_S = LamGE_S
        self.LamGE_T = LamGE_T
        self.LamCTG = 0.e0
        self.LamECT = 0.e0
        self.LamCTpT= 0.e0
        self.LamCTnT= 0.e0


    def is_neighbor(self, a, b, periodic=True):
        """True if sites a and b are nearest neighbors on the 1D chain,
        including the periodic boundary when enabled.
        """
        if abs(a - b) == 1:
            return True
        if periodic and self.Nchrom > 2 and abs(a - b) == self.Nchrom - 1:
            return True
        return False
 
    def site_distance(self, i, j, periodic=False):
        """Distance between chromophore indices in units of nearest-neighbor spacing.

        If periodic=False:
            distance = abs(i - j)

        If periodic=True:
            use the shortest distance around the ring.
        """
        if i == j:
            return 0

        d = abs(i - j)

        if periodic:
            d = min(d, self.Nchrom - d)

        return d


    def calc_JCou(self, i, j, periodic=False, decay_power=1.0):
        """Distance-dependent Coulombic Frenkel coupling.

        For nearest neighbors:
            distance = 1, so J = JCou_inter.

        For farther pairs:
            J = JCou_inter / distance**decay_power.

        decay_power = 1.0 gives J/R.
        decay_power = 3.0 gives dipole-dipole-like J/R^3.
        """
        d = self.site_distance(i, j, periodic=periodic)

        if d == 0:
            return 0.0

        return self.JCou_inter / (d ** decay_power)


##generating Hamiltonian matrix block by block
    def gen_Ham ( self ):
        ### 0.1, initializing the parameters and calculating some nessicery parameters
        #(a) energy levels
        Ham_EL = EL()
        self.E_CT_PpPn = Ham_EL.calcEPpPn()
        self.E_CT_PnPp = Ham_EL.calcEPnPp()
        self.E_TP = Ham_EL.calcEPtPt()
        #(b) potential well shift value
        Ham_HR = HR_fac()
        self.LamCTG = Ham_HR.calculate_LamCTG()
        self.LamECT = Ham_HR.calculate_LamECT()
        self.LamCTpT= Ham_HR.calculate_LamCTpT()
        self.LamCTnT= Ham_HR.calculate_LamCTnT()
        ###0.2, INdexing basis set 
        print("Start Indexing Basis Set")
        start_time = tm.time()
        Ham_IBS = IBS()
        self.Index_single = Ham_IBS.arr_1p()
        num_1p = np.count_nonzero( self.Index_single )
        print(f'The number of 1-particle basis is ', num_1p + 1 )
        if self.add_double:
            self.Index_double = Ham_IBS.arr_2p()
        else:
            print("We did not include 2-particle basis set")
        if self.add_tripple:
            self.Index_tripple = Ham_IBS.arr_3p()
            num_3p = np.count_nonzero( self.Index_tripple )
            print(f'The number of 3-particle basis is ', num_3p)
        else:
            print("We did not include 3-particle basis set")
        if self.add_CT:
            self.Index_CT = Ham_IBS.arr_CT()
            num_CT = np.count_nonzero( self.Index_CT )
            print(f'The number of CT basis is ', num_CT )
        else:
            print("We did not include CT basis set")
        if self.add_CTv:
            self.Index_CTv = Ham_IBS.arr_CTv()
            num_CTv = np.count_nonzero( self.Index_CTv )
            print(f'The number of CTv basis is ', num_CTv )
        else:
            print("We did not include CTv basis set")
        if self.add_TP:
            self.Index_TP = Ham_IBS.arr_TP()
            num_TP = np.count_nonzero( self.Index_TP )
            print(f'The number of TP basis is ' , num_TP)
        else:
            print("We did not include Triplet Pair basis set") 
        if self.add_TPv:
            self.Index_TPv = Ham_IBS.arr_TPv()
            num_TPv = np.count_nonzero( self.Index_TPv )
            print(f'The number of TPv basis is ' , num_TPv )
        else:
            print("We did not include Triplet Pair with vibration basis set")   
        end_time = tm.time()
        ##Now, all basis sets are indexed. We need to read the total number of basis set
        self.kcount = Ham_IBS.kcount  ##important
        print(f'The total dimension of the Hamiltonian is ', self.kcount)

        print("Indexing basis set done")
        ts = end_time - start_time
        print("Time consumed by indexing basis set is", ts )

        ###0.3,Then, the vibrational energy is in constant unit
        print(f'Start generating Hamiltonian')
        self.ESex_ref = self.ESex_ref * self.eV / self.vib_freq
        self.E_CT_PpPn = self.E_CT_PpPn * self.eV / self.vib_freq
        self.E_CT_PnPp = self.E_CT_PnPp * self.eV / self.vib_freq
        self.E_TP = self.E_TP * self.eV / self.vib_freq
        self.JCou_inter = self.JCou_inter * self.eV / self.vib_freq
        self.te_inter_t = self.te_inter_t * self.eV / self.vib_freq
        self.th_inter_t = self.th_inter_t * self.eV / self.vib_freq
        self.te_inter_s = self.te_inter_s * self.eV / self.vib_freq
        self.th_inter_s = self.th_inter_s * self.eV / self.vib_freq
        self.K_exchange = self.K_exchange * self.eV / self.vib_freq
        self.VMU_inter = self.VMU_inter * self.eV / self.vib_freq

        ### 0.4 include the FC factor function
        Ham_FCF = FCF()

        ### 0.5 Initializing the Hamilitonian !!!!
        self.diabatic_Ham = np.zeros(( self.kcount, self.kcount ) , dtype=float )

        ###1. Diagonal block
        #1.1, 1 particle block in singlet exciton
        for i in range( 0 , self.Nchrom  ):
            for i1 in range( 0 , self.vibmax + 1 ):
                num_1p = Ham_IBS.order_1p( i , i1 )
                lab1 = self.Index_single[num_1p ]
                for l in range( 0 , self.Nchrom  ):
                    for l1 in range( 0 , self.vibmax + 1 ):
                        num_1p = Ham_IBS.order_1p( l , l1 )
                        lab2 = self.Index_single[num_1p]  
                        if ( lab1 == lab2 ): ##diagonal elements
                            self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.ESex_ref + np.multiply( i1 , 1.0 , dtype=float )
                        else:  ##off-diagonal elements
                            _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                            self.diabatic_Ham[ ( lab1, lab2 ) ]  = _tem_JCou * Ham_FCF.gen_FCF( 0, i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0, l1, self.LamGE_S )
                            self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
        #1.2, 2 particle block in singlet exciton
        if self.add_double:
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            if  i == j:
                             continue
                            else:
                                if i1 + j1 + 1 > self.vibmax:
                                    continue
                                else:
                                    num_2p = Ham_IBS.order_2p(i, i1, j, j1)
                                    lab1 = self.Index_double[num_2p]  
                                    for l in range( 0 , self.Nchrom ):
                                        for l1 in range( 0 , self.vibmax + 1 ):
                                            for m in range( 0 , self.Nchrom ):
                                                for m1 in range( 0 , self.vibmax ):
                                                    if  l == m:
                                                        continue
                                                    else:
                                                        if l1 + m1 + 1 > self.vibmax:
                                                            continue
                                                        else:
                                                            num_2p = Ham_IBS.order_2p( l, l1, m, m1 )
                                                            lab2 = self.Index_double[num_2p]
                                                            if (lab1 == lab2): ## diagonal elements
                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  =  self.ESex_ref + Sfactor * l1 + Sfactor * (m1 + 1)
                                                            else: ##off-diagonal elements
                                                                if ( i == m and j == l ):
                                                                    _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                    self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( m1 + 1 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( j1 + 1, l1, self.LamGE_S )
                                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                elif ( j == m and j1 == m1 ):
                                                                    _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                    self.diabatic_Ham[ (lab1, lab2 )]  = _tem_JCou * Ham_FCF.gen_FCF( 0, i1 , self.LamGE_S ) * Ham_FCF.gen_FCF( 0, l1 , self.LamGE_S )
                                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                else:
                                                                    self.diabatic_Ham[ (lab1, lab2 )]  = 0.e0
        else:
            print(f'We did not include 2-particle basis set')
        ##1.3, 3 particle block in singlet exciton
        if self.add_tripple :
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            for k in range( 0 , self.Nchrom ):
                                for k1 in range( 0 , self.vibmax ):
                                    if ( i == j or  j >=  k or  k == i ):
                                        continue
                                    else:
                                        if i1 + j1 + 1 + k1 + 1 > self.vibmax:
                                            continue
                                        else:
                                            if ( self.is_neighbor(i, j, self.JCou_inter_periodic) or self.is_neighbor(i, k, self.JCou_inter_periodic) ):
                                                num_3p = Ham_IBS.order_3p(  i, i1, j, j1, k, k1 )
                                                lab1 = self.Index_tripple[ num_3p ]
                                                for l in range( 0 , self.Nchrom ):
                                                    for l1 in range( 0 , self.vibmax + 1 ):
                                                        for m in range( 0 , self.Nchrom ):
                                                            for m1 in range( 0 , self.vibmax ):
                                                                for n in range( 0 , self.Nchrom ):
                                                                    for n1 in range( 0 , self.vibmax ):
                                                                        if ( l == m or  m >=  n or  n == l ):
                                                                            continue
                                                                        else:
                                                                            if l1 + m1 + 1 + n1 + 1 > self.vibmax:
                                                                                continue
                                                                            else:
                                                                                if ( self.is_neighbor(l, m, self.JCou_inter_periodic) or self.is_neighbor(l, n, self.JCou_inter_periodic) ):
                                                                                    num_3p = Ham_IBS.order_3p(  l, l1, m, m1, n, n1 )
                                                                                    lab2 = self.Index_tripple[ num_3p ]
                                                                                    if (lab1 == lab2): ## diagonal elements
                                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.ESex_ref + Sfactor * l1 + Sfactor * (m1 + 1) + Sfactor * (n1 + 1)
                                                                                    else: #off-diagonal elements
                                                                                        if (i != l):
                                                                                            if (j == l and i == m and k == n and k1 == n1):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( m1 + 1, i1, self.LamGE_S ) * Ham_FCF.gen_FCF( j1 +1, l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif(j == l and i == n and k == m and m1 == k1):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( n1 + 1, i1, self.LamGE_S ) * Ham_FCF.gen_FCF( j1 +1, l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif(k == l and i == m and j == n and j1 == n1 ):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( m1 + 1, i1, self.LamGE_S ) * Ham_FCF.gen_FCF( k1 +1, l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif(k == l and i == n and j == m and j1 == m1):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( n1 + 1, i1, self.LamGE_S ) * Ham_FCF.gen_FCF( k1 +1, l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif(j == m and j1 == m1 and k == n and k1 == n1):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( 0 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0 , l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif( j == n and j1 == n1 and k == m and k1 == m1 ):
                                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = _tem_JCou * Ham_FCF.gen_FCF( 0 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0 , l1, self.LamGE_S )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            else:
                                                                                                self.diabatic_Ham[ (lab1, lab2 )] = 0.e0
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]

        else:
            print(f'We did not include 3-particle basis set')
    #1.4, CT block
        if self.add_CT:
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            if i == j:
                                continue
                            else:
                                if( self.is_neighbor(i, j, self.CT_inter_periodic ) ):
                                    if ( i1 + j1 > self.vibmax ):
                                        continue
                                    else:
                                        num_CT = Ham_IBS.order_CT( i, i1, j, j1 )
                                        lab1 = self.Index_CT[num_CT] 
                                        for l in range( 0 , self.Nchrom ):
                                            for l1 in range( 0 , self.vibmax + 1 ):
                                                for m in range( 0 , self.Nchrom ):
                                                    for m1 in range( 0 , self.vibmax + 1 ):
                                                        if l == m:
                                                            continue
                                                        else:
                                                            if( self.is_neighbor(l, m, self.CT_inter_periodic ) ):
                                                                if ( l1 + m1 > self.vibmax ):
                                                                    continue
                                                                else:
                                                                    num_CT = Ham_IBS.order_CT( l, l1, m, m1 )
                                                                    lab2 = self.Index_CT[num_CT]
                                                                    if( lab1 == lab2 ):      ## diagonal elements
                                                                        if ( i < j ):
                                                                            self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.E_CT_PpPn + np.multiply(  self.Sfactor ,l1 , dtype=float ) + np.multiply(  self.Sfactor ,m1 , dtype=float ) 
                                                                        elif( i > j ):
                                                                            self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.E_CT_PnPp + np.multiply(  self.Sfactor ,l1 , dtype=float ) + np.multiply(  self.Sfactor ,m1 , dtype=float )
                                                                    else: ## off-diagonal elements 
                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
                                                                        
        else:
            print(f'We did not include CT basis set')
    #1.5, CTv block
        if self.add_CTv :
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            for k in range( 0 , self.Nchrom ):
                                for k1 in range( 0 , self.vibmax ):
                                    if (  i == j or j == k or i  == k ) :
                                        continue
                                    else:
                                        if (self.is_neighbor( i, j , self.CT_inter_periodic ) ):
                                            if ( i1 + j1 + k1 + 1 > self.vibmax ):
                                                continue
                                            else:
                                                if ( self.is_neighbor( i, k , self.CT_inter_periodic ) or self.is_neighbor( j, k , self.CT_inter_periodic ) ): 
                                                    num_CTv = Ham_IBS.order_CTv( i, i1, j, j1, k, k1 )
                                                    lab1 = self.Index_CTv[num_CTv]
                                                    for l in range( 0 , self.Nchrom ):
                                                        for l1 in range( 0 , self.vibmax + 1 ):
                                                            for m in range( 0 , self.Nchrom ):
                                                                for m1 in range( 0 , self.vibmax + 1 ):
                                                                    for n in range( 0 , self.Nchrom ):
                                                                        for n1 in range( 0 , self.vibmax ):
                                                                            if (  l == m or m == n or l  == n ) :
                                                                                continue
                                                                            else:
                                                                                if ( self.is_neighbor( l, m , self.CT_inter_periodic )  ):
                                                                                    if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                                        continue
                                                                                    else:
                                                                                        if ( self.is_neighbor( l, n , self.CT_inter_periodic ) or self.is_neighbor( m, n , self.CT_inter_periodic ) ): 
                                                                                            num_CTv = Ham_IBS.order_CTv( l, l1, m, m1, n, n1 )
                                                                                            lab2 = self.Index_CTv[num_CTv]
                                                                                            if (lab1 == lab2):##diagonal elements
                                                                                                if ( i < j ):
                                                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.E_CT_PpPn + Sfactor * l1 + Sfactor * m1 + Sfactor * (n1 + 1)
                                                                                                elif( i > j ):
                                                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.E_CT_PnPp + Sfactor * l1 + Sfactor * m1 + Sfactor * (n1 + 1) 
                                                                                            else: ##off-diagonal elements
                                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include CTv basis set')    
    ##1.6, Triplet pair block
        if self.add_TP :
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            if i >= j:
                                continue
                            else:
                                if( self.is_neighbor( i, j, self.CT_inter_periodic ) ):
                                    if ( i1 + j1 > self.vibmax ):
                                        continue
                                    else:
                                        num_TP = Ham_IBS.order_TP( i, i1, j, j1 )
                                        lab1 = self.Index_TP[num_TP] 
                                        for l in range( 0 , self.Nchrom ):
                                            for l1 in range( 0 , self.vibmax + 1 ):
                                                for m in range( 0 , self.Nchrom ):
                                                    for m1 in range( 0 , self.vibmax + 1 ):
                                                        if l >= m:
                                                            continue
                                                        else:
                                                            if( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                if ( l1 + m1 > self.vibmax ):
                                                                    continue
                                                                else:
                                                                    num_TP = Ham_IBS.order_TP( l, l1, m, m1 )
                                                                    lab2 = self.Index_TP[num_TP]
                                                                    if( lab1 == lab2 ):      ## diagonal elements
                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.E_TP + np.multiply(  self.Sfactor ,l1 , dtype=float ) + np.multiply( self.Sfactor ,m1 , dtype=float )
                                                                    else: ## off-diagonal elements 
                                                                        self.diabatic_Ham[ (lab1, lab2 ) ]  = 0.00
                                                                        self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]

        else:
            print(f'We did not include Triplet pair basis set') 
    ##1.7, Triplet pair with vibration block
        if self.add_TPv :
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            for k in range( 0 , self.Nchrom ):
                                for k1 in range( 0 , self.vibmax ):
                                    if (  i >= j or j == k or i  == k ) :
                                        continue
                                    else:
                                        if ( self.is_neighbor( i, j, self.CT_inter_periodic ) ):
                                            if ( i1 + j1 + k1 + 1 > self.vibmax ):
                                                continue
                                            else:
                                                if self.is_neighbor( i, k , self.CT_inter_periodic) or self.is_neighbor( j, k , self.CT_inter_periodic ): 
                                                    num_TPv = Ham_IBS.order_TPv( i, i1, j, j1, k, k1 )
                                                    lab1 = self.Index_TPv[num_TPv]
                                                    for l in range( 0 , self.Nchrom ):
                                                        for l1 in range( 0 , self.vibmax + 1 ):
                                                            for m in range( 0 , self.Nchrom ):
                                                                for m1 in range( 0 , self.vibmax + 1 ):
                                                                    for n in range( 0 , self.Nchrom ):
                                                                        for n1 in range( 0 , self.vibmax ):
                                                                            if (  l >= m or m == n or l  == n ) :
                                                                                continue
                                                                            else: 
                                                                                if ( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                                    if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                                        continue
                                                                                    else:
                                                                                        if ( self.is_neighbor( l, n , self.CT_inter_periodic) or self.is_neighbor( m, n , self.CT_inter_periodic) ):
                                                                                            if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                                                continue
                                                                                            else: 
                                                                                                num_TPv = Ham_IBS.order_TPv( l, l1, m, m1, n, n1 )
                                                                                                lab2 = self.Index_TPv[num_TPv]
                                                                                                if (lab1 == lab2):##diagonal elements
                                                                                                    self.diabatic_Ham[(lab1, lab2)] = self.E_TP + Sfactor * l1 + Sfactor * m1 + Sfactor * (n1 + 1)
                                                                                                else: ##off-diagonal elements
                                                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
                                                                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
        else:
            print(f'We did not include Triplet pair with vibration basis set')
        ##Diagonal blocks are done
        ###2. Off-diagonal BLOCK
        ##2.1, coupling between 1P and 2P states.
        if self.add_double:
            for i in range( 0 , self.Nchrom  ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    num_1p = Ham_IBS.order_1p( i , i1 )
                    lab1 = self.Index_single[num_1p]  
                    for l in range( 0 , self.Nchrom ):
                        for l1 in range( 0 , self.vibmax + 1 ):
                            for m in range( 0 , self.Nchrom ):
                                for m1 in range( 0 , self.vibmax ):
                                    if  l == m:
                                        continue
                                    else:
                                        if l1 + m1 + 1 > self.vibmax:
                                            continue
                                        else:
                                            num_2p = Ham_IBS.order_2p( l, l1, m, m1 )
                                            lab2 = self.Index_double[num_2p]
                                            if ( i == m ):
                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = _tem_JCou * Ham_FCF.gen_FCF( m1 + 1 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0 , l1, self.LamGE_S )
                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                            else:
                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include 2 particle basis set and then the off-diagonal block is zero')
        ##2.2, coupling between 1P and 3P states are all zero, due to two vibration states in 3P 

        ##2.3, coupling between 2P and 3P states.
        if (self.add_double and self.add_tripple) :       
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            if  i == j:
                             continue
                            else:
                                if i1 + j1 + 1 > self.vibmax:
                                    continue
                                else:
                                    num_2p = Ham_IBS.order_2p( i, i1, j, j1 )
                                    lab1 = self.Index_double[num_2p]  
                                    for l in range( 0 , self.Nchrom ):
                                        for l1 in range( 0 , self.vibmax + 1 ):
                                            for m in range( 0 , self.Nchrom ):
                                                for m1 in range( 0 , self.vibmax ):
                                                    for n in range( 0 , self.Nchrom ):
                                                        for n1 in range( 0 , self.vibmax ):
                                                            if ( l == m or  m >=  n or  n == l ):
                                                                continue
                                                            else:
                                                                if l1 + m1 + 1 + n1 + 1 > self.vibmax:
                                                                    continue
                                                                else:
                                                                    if ( self.is_neighbor(l, m, self.JCou_inter_periodic) or self.is_neighbor(l, n, self.JCou_inter_periodic) ):
                                                                        num_3p = Ham_IBS.order_3p(  l, l1, m, m1, n, n1 )
                                                                        lab2 = self.Index_tripple[ num_3p ]
                                                                        if ( i != l):
                                                                            if ( i == m and j == n and j1== n1 ):
                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  =  _tem_JCou * Ham_FCF.gen_FCF( m1 + 1 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0 , l1, self.LamGE_S )
                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                            elif( i == n and j == m and j1== m1 ):
                                                                                _tem_JCou =  self.calc_JCou( i, l, periodic=self.JCou_inter_periodic, decay_power=1.0)
                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  =  _tem_JCou * Ham_FCF.gen_FCF( n1 + 1 , i1, self.LamGE_S ) * Ham_FCF.gen_FCF( 0 , l1, self.LamGE_S )
                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                        else:
                                                                            self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include 2 particle nor 3 particle basis set and then the off-diagonal block is zero')
        ## 2.4, coupling between 1P and CT states.
        if ( self.add_CT ):
            for i in range( 0 , self.Nchrom  ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    num_1p = Ham_IBS.order_1p( i , i1 )
                    lab1 = self.Index_single[num_1p ]
                    for l in range( 0 , self.Nchrom ):
                        for l1 in range( 0 , self.vibmax + 1 ):
                            for m in range( 0 , self.Nchrom ):
                                for m1 in range( 0 , self.vibmax + 1 ):
                                    if l == m:
                                        continue
                                    else:
                                        if( self.is_neighbor(l, m, self.CT_inter_periodic ) ):
                                            if ( l1 + m1 > self.vibmax ):
                                                continue
                                            else:
                                                num_CT = Ham_IBS.order_CT( l, l1, m, m1 )
                                                lab2 = self.Index_CT[num_CT]
                                                if( i == l ):
                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.te_inter_s * Ham_FCF.gen_FCF( l1  , i1 , self.LamECT ) * Ham_FCF.gen_FCF( 0 , m1 , self.LamCTG )
                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                elif( i == m ):
                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.th_inter_s * Ham_FCF.gen_FCF( m1 , i1 , self.LamECT ) * Ham_FCF.gen_FCF( 0 , l1 , self.LamCTG )
                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                else:
                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include CT basis set and then the off-diagonal block is zero')
        ## 2.5, coupling between 2P and CT states.
        if ( self.add_CT and self.add_double ):
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            if  i == j:
                             continue
                            else:
                                if i1 + j1 + 1 > self.vibmax:
                                    continue
                                else:
                                    num_2p = Ham_IBS.order_2p( i, i1, j, j1 )
                                    lab1 = self.Index_double[num_2p]  
                                    for l in range( 0 , self.Nchrom ):
                                        for l1 in range( 0 , self.vibmax + 1 ):
                                            for m in range( 0 , self.Nchrom ):
                                                for m1 in range( 0 , self.vibmax + 1 ):
                                                    if l == m:
                                                        continue
                                                    else:
                                                        if( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                            if ( l1 + m1 > self.vibmax ):
                                                                continue
                                                            else:
                                                                num_CT = Ham_IBS.order_CT( l, l1, m, m1 )
                                                                lab2 = self.Index_CT[num_CT]
                                                                if( i == l and j == m ):
                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.te_inter_s * Ham_FCF.gen_FCF( l1 , i1 , self.LamECT ) * Ham_FCF.gen_FCF( j1 + 1, m1 , self.LamCTG )
                                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                elif( i == m and j == l ):
                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.th_inter_s * Ham_FCF.gen_FCF( m1  , i1 , self.LamECT ) * Ham_FCF.gen_FCF( j1 + 1, l1, self.LamCTG )
                                                                    self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                else:
                                                                    self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include 2P basis set nor CT basis set and then the off-diagonal block is zero')
        ## 2.6, coupling between 3P and CT states are all zero.

        ## 2.7, coupling between 1P and CTv states are all zero, due to two vibrations variables in CTv

        ## 2.8, coupling between 2P and CTv states.
        if ( self.add_CTv and self.add_double ):
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            if  i == j:
                             continue
                            else:
                                if i1 + j1 + 1 > self.vibmax:
                                    continue
                                else:
                                    num_2p = Ham_IBS.order_2p( i, i1, j, j1 )
                                    lab1 = self.Index_double[num_2p]  
                                    for l in range( 0 , self.Nchrom ):
                                        for l1 in range( 0 , self.vibmax + 1 ):
                                            for m in range( 0 , self.Nchrom ):
                                                for m1 in range( 0 , self.vibmax + 1 ):
                                                    for n in range( 0 , self.Nchrom ):
                                                        for n1 in range( 0 , self.vibmax ):
                                                            if (  l == m or m == n or l  == n ) :
                                                                continue
                                                            else:
                                                                if ( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                    if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                        continue
                                                                    else:
                                                                        if ( self.is_neighbor( l, n , self.CT_inter_periodic) or self.is_neighbor( m, n , self.CT_inter_periodic) ): 
                                                                            num_CTv = Ham_IBS.order_CTv( l, l1, m, m1, n, n1 )
                                                                            lab2 = self.Index_CTv[num_CTv]
                                                                            if ( i == l and j== n and j1 == n1 ):
                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.te_inter_s * Ham_FCF.gen_FCF( l1  , i1 , self.LamECT ) * Ham_FCF.gen_FCF( 0 , m1  , self.LamCTG )
                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                            elif( i == m and j == n and j1 == n1 ):
                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = self.th_inter_s * Ham_FCF.gen_FCF( m1  , i1  , self.LamECT ) * Ham_FCF.gen_FCF( 0 , l1 , self.LamCTG )
                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                            else:
                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0
        else:
            print(f'We did not include 2P basis set or CT basis set and then the off-diagonal block is zero')
        ## 2.9, coupling between 3P and CTv states 
        if self.add_tripple and self.add_CTv:
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax ):
                            for k in range( 0 , self.Nchrom ):
                                for k1 in range( 0 , self.vibmax ):
                                    if ( i == j or  j >=  k or  k == i ):
                                        continue
                                    else:
                                        if i1 + j1 + 1 + k1 + 1 > self.vibmax:
                                            continue
                                        else:
                                            if ( self.is_neighbor(i, j, self.JCou_inter_periodic) or self.is_neighbor(i, k, self.JCou_inter_periodic) ):
                                                num_3p = Ham_IBS.order_3p(  i, i1, j, j1, k, k1 )
                                                lab1 = self.Index_tripple[ num_3p ]
                                                for l in range( 0 , self.Nchrom ):
                                                    for l1 in range( 0 , self.vibmax + 1 ):
                                                        for m in range( 0 , self.Nchrom ):
                                                            for m1 in range( 0 , self.vibmax + 1 ):
                                                                for n in range( 0 , self.Nchrom ):
                                                                    for n1 in range( 0 , self.vibmax ):
                                                                        if (  l == m or m == n or l  == n ) :
                                                                            continue
                                                                        else:
                                                                            if ( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                                if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                                    continue
                                                                                else:
                                                                                    if ( self.is_neighbor( l, n , self.CT_inter_periodic) or self.is_neighbor( m, n , self.CT_inter_periodic) ): 
                                                                                        num_CTv = Ham_IBS.order_CTv( l, l1, m, m1, n, n1 )
                                                                                        lab2 = self.Index_CTv[num_CTv]
                                                                                        if ( i == l ):
                                                                                            if ( k == n and j == m and k1 == n1 ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = self.te_inter_s * Ham_FCF.gen_FCF( l1  , i1  , self.LamECT ) * Ham_FCF.gen_FCF( j1 + 1 , m1 , self.LamCTG )
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )] 
                                                                                            elif ( j == n and k == m and j1 == n1 ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = self.te_inter_s * Ham_FCF.gen_FCF( l1  , i1  , self.LamECT ) * Ham_FCF.gen_FCF( k1 + 1 , m1 , self.LamCTG )
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )] 
                                                                                            else:
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = 0.e0
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )] 
                                                                                        elif ( i == m ):
                                                                                            if ( j == l and k == n and k1 == n1 ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = self.th_inter_s * Ham_FCF.gen_FCF( m1  , i1  , self.LamECT ) * Ham_FCF.gen_FCF( j1 + 1 , l1 , self.LamCTG )
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif ( j == n and j1 == n1 and k == l ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = self.th_inter_s * Ham_FCF.gen_FCF( m1  , i1  , self.LamECT ) * Ham_FCF.gen_FCF( k1 + 1 , l1 , self.LamCTG )
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            else:
                                                                                                self.diabatic_Ham[ ( lab1, lab2 )] = 0.e0
                                                                                                self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )] 
                                                                                        else:
                                                                                            self.diabatic_Ham[ ( lab1, lab2 )] = 0.e0
                                                                                            self.diabatic_Ham[ (lab2, lab1 ) ] = self.diabatic_Ham[ (lab1, lab2 )] 
        else:
            print(f'We did not include 3P basis set or CTv basis set and then the off-diagonal block is zero')
###!!!!Triplet pair states only couple with CT states, since singlets and triplets are orthogonal to each other in terms of symmetry in electronic states !!!!
        ## 2.10, coupling between TP and CT states.
        if ( self.add_CT and self.add_TP ):
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            if i == j:
                                continue
                            else:
                                if( self.is_neighbor( i, j, self.CT_inter_periodic ) ):
                                    if ( i1 + j1 > self.vibmax ):
                                        continue
                                    else:
                                        num_CT = Ham_IBS.order_CT( i, i1, j, j1 )
                                        lab1 = self.Index_CT[num_CT] 
                                        for l in range( 0 , self.Nchrom ):
                                            for l1 in range( 0 , self.vibmax + 1 ):
                                                for m in range( 0 , self.Nchrom ):
                                                    for m1 in range( 0 , self.vibmax + 1 ):
                                                        if l >= m:
                                                            continue
                                                        else:
                                                            if( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                if ( l1 + m1 > self.vibmax ):
                                                                    continue
                                                                else:
                                                                    num_TP = Ham_IBS.order_TP( l, l1, m, m1 )
                                                                    lab2 = self.Index_TP[num_TP]
                                                                    if ( i == l and j == m ):
                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = np.sqrt(3/2) * self.te_inter_t * Ham_FCF.gen_FCF( l1 , i1 , self.LamCTpT ) * Ham_FCF.gen_FCF( j1 , m1 , self.LamCTnT )
                                                                        self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                    elif( i == m and j == l ):
                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = np.sqrt(3/2) * self.th_inter_t * Ham_FCF.gen_FCF( m1 , i1 , self.LamCTpT ) * Ham_FCF.gen_FCF( j1 , l1 , self.LamCTnT )
                                                                        self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                    else:
                                                                        self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0

        else:
            print(f'We did not include TP basis set nor CT basis set and then the off-diagonal block is zero')

###!!!Also, notice that the vibration state in TPv or CTv is also electronical ground state, then there is no coupling between TP with vibrational state
        ## 2.11, coupling between TPv and CTv states.
        if ( self.add_CTv and self.add_TPv ):
            for i in range( 0 , self.Nchrom ):
                for i1 in range( 0 , self.vibmax + 1 ):
                    for j in range( 0 , self.Nchrom ):
                        for j1 in range( 0 , self.vibmax + 1 ):
                            for k in range( 0 , self.Nchrom ):
                                for k1 in range( 0 , self.vibmax ):
                                    if (  i == j or j == k or i  == k ) :
                                        continue
                                    else:
                                        if ( self.is_neighbor( i, j, self.CT_inter_periodic ) ):
                                            if ( i1 + j1 + k1 + 1 > self.vibmax ):
                                                continue
                                            else:
                                                if (  self.is_neighbor( i, k , self.CT_inter_periodic) or self.is_neighbor( j, k , self.CT_inter_periodic) ): 
                                                    num_CTv = Ham_IBS.order_CTv( i, i1, j, j1, k, k1 )
                                                    lab1 = self.Index_CTv[num_CTv]
                                                    for l in range( 0 , self.Nchrom ):
                                                        for l1 in range( 0 , self.vibmax + 1 ):
                                                            for m in range( 0 , self.Nchrom ):
                                                                for m1 in range( 0 , self.vibmax + 1 ):
                                                                    for n in range( 0 , self.Nchrom ):
                                                                        for n1 in range( 0 , self.vibmax ):
                                                                            if (  l >= m or m == n or l  == n ) :
                                                                                continue
                                                                            else:
                                                                                if ( self.is_neighbor( l, m, self.CT_inter_periodic ) ):
                                                                                    if ( l1 + m1 + n1 + 1 > self.vibmax ):
                                                                                        continue
                                                                                    else:
                                                                                        if (  self.is_neighbor( l, n , self.CT_inter_periodic) or self.is_neighbor( m, n , self.CT_inter_periodic) ): 
                                                                                            num_TPv = Ham_IBS.order_TPv( l, l1, m, m1, n, n1 )
                                                                                            lab2 = self.Index_TPv[num_TPv]
                                                                                            if (  i == l and j == m and k == n and k1 == n1 ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = np.sqrt(3/2) * self.te_inter_t * Ham_FCF.gen_FCF( l1  , i1 , self.LamCTpT ) * Ham_FCF.gen_FCF( j1  , m1 , self.LamCTnT )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            elif( i == m and j == l and k == n and k1 == n1 ):
                                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = np.sqrt(3/2) * self.th_inter_t * Ham_FCF.gen_FCF( m1  , i1 , self.LamCTpT ) * Ham_FCF.gen_FCF( j1  , l1 , self.LamCTnT )
                                                                                                self.diabatic_Ham[ (lab2, lab1 )] = self.diabatic_Ham[ (lab1, lab2 )]
                                                                                            else:
                                                                                                self.diabatic_Ham[ ( lab1, lab2 ) ]  = 0.e0

        else:
            print(f'We did not include TPv basis set nor CTv basis set and then the off-diagonal block is zero')

        return self.diabatic_Ham , self.kcount
