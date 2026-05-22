"""Oscillator strengths for single-exciton transitions from the ground state.

Only 1-particle (1P) basis states carry transition dipole moment from the
electronic ground state, because the ground state has zero vibrational quanta
on every site and is electronically closed-shell. 2P/3P/CT/TP basis states all
involve at least one electronically-excited or vibrationally-excited spectator
that has zero overlap with the ground state.

So the transition dipole of eigenstate |alpha> is:

    mu_alpha = sum_{l, l1} <l, l1 | alpha> * mu_chrom(l) * <0_vib | l1>_FC

where mu_chrom(l) is the molecular transition dipole of chromophore l (taken
identical in magnitude across sites, with orientation set by `theta`), and
<0|l1>_FC = gen_FCF(0, l1, lambda) is the vibrational overlap between the
ground-state vibrational wavefunction and the displaced excited-state |l1>.

The oscillator strength is f_alpha = (E_alpha - E_0) * |mu_alpha|^2, up to
unit prefactors. With E_0 = 0 in this code (ground state), that becomes
f_alpha proportional to E_alpha * |mu_alpha|^2.
"""

import json
import numpy as np

from basis_set import IBS
from FC_factor import FCF


with open('parameters.json') as f:
    _params = json.load(f)

_NCHROM   = _params['geometry_parameters']['Nchrom']
_VIBMAX   = _params['geometry_parameters']['vibmax']
_THETA    = _params['geometry_parameters']['theta']         # degrees
_LAM_GE_S = _params['huang_ryhs_factors']['LamGE_S']
_E_EX_S   = _params['Energy_setting']['E_ex_s']
_FREQ_FAC = _params['Abs_plotting']['abs_freq_fac_switch']


class OSC:
    """Compute oscillator strengths from a diagonalized Frenkel-CT Hamiltonian.

    Parameters
    ----------
    kcount : int
        Total dimension of the Hamiltonian / number of eigenstates.
    evect : (kcount, kcount) ndarray
        Eigenvectors, column `alpha` is eigenstate alpha.
    evalue : (kcount,) ndarray
        Eigenvalues in eV (already converted in Dia_EE).
    """

    def __init__(self, kcount, evect, evalue):
        self.kcount = kcount
        self.evect  = evect
        self.evalue = evalue

        # parameters
        self.Nchrom    = _NCHROM
        self.vibmax    = _VIBMAX
        self.theta_deg = _THETA              # kept in degrees, never mutated
        self.lam       = _LAM_GE_S
        self.E_ex_s    = _E_EX_S
        self.freq_fac  = _FREQ_FAC

        # 1P basis indexing — built once, used by all methods
        ibs = IBS()
        self.Index_single = ibs.arr_1p()

        # FC overlaps <0|l1> for l1 = 0..vibmax — cached once
        fcf = FCF()
        self._fc0 = np.array(
            [fcf.gen_FCF(0, l1, self.lam) for l1 in range(self.vibmax + 1)],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cal_OSC(self, save_path=None):
        """Oscillator strengths for all eigenstates.

        Returns
        -------
        OSC_X, OSC_Y : (kcount,) ndarrays
            x- and y-polarized oscillator strengths.
        """
        TDM_X, TDM_Y = self._transition_dipoles()
        OSC_X, OSC_Y = self._dipoles_to_oscillator(TDM_X, TDM_Y)

        if save_path is not None:
            np.savetxt(save_path, np.column_stack([OSC_X, OSC_Y]),
                       fmt='%2.6f', header='OSC_X  OSC_Y')

        print('Oscillator strengths computed for {} eigenstates'
              .format(self.kcount))
        return OSC_X, OSC_Y

    def character_weights(self, block_sizes):
        """Decompose each eigenstate into basis-block character weights.

        Parameters
        ----------
        block_sizes : dict[str, tuple[int, int]]
            Maps block name -> (start, end) row index in the eigenvector.
            E.g. {'1P': (0, 20), '2P': (20, 200), 'CT': (200, 380), ...}.

        Returns
        -------
        weights : dict[str, (kcount,) ndarray]
            For each block, weights[block][alpha] = sum_{i in block} |evect[i, alpha]|^2.
        """
        psi2 = self.evect ** 2                                  # |c_i^alpha|^2
        return {name: psi2[start:end, :].sum(axis=0)
                for name, (start, end) in block_sizes.items()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition_dipoles(self):
        """Construct mu_X, mu_Y for every eigenstate via a single matvec.

        Builds the (kcount,)-shaped projection vector `proj` such that
            proj[lab(l, l1)] = FC(0, l1)
        for every 1P basis state (l, l1), zero elsewhere. Then
            mu_X[alpha] = cos(theta) * (proj @ evect)[alpha]
            mu_Y[alpha] = sin(theta) * (proj @ evect)[alpha]
        """
        theta_rad = np.deg2rad(self.theta_deg)

        proj = np.zeros(self.kcount, dtype=float)
        for l in range(self.Nchrom):
            for l1 in range(self.vibmax + 1):
                row_idx = self.Index_single[(l) * (self.vibmax + 1) + l1]
                proj[row_idx] = self._fc0[l1]

        # proj has shape (kcount,); evect has shape (kcount, kcount).
        # Result has shape (kcount,), one transition dipole per eigenstate.
        mu = proj @ self.evect

        return np.cos(theta_rad) * mu, np.sin(theta_rad) * mu

    def _dipoles_to_oscillator(self, TDM_X, TDM_Y):
        if self.freq_fac:
            energy = self.E_ex_s + self.evalue                  # transition energy in eV
            return energy * TDM_X ** 2, energy * TDM_Y ** 2
        return TDM_X ** 2, TDM_Y ** 2