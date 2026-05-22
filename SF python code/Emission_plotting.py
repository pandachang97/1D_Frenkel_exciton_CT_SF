"""Emission spectrum from a diagonalized Frenkel-CT-TT Hamiltonian.

Each eigenstate alpha is weighted by a Boltzmann population with the partition
function summed over ALL eigenstates (no spin filtering, no "find the singlet"
step). At T = 0, all population sits in the absolute lowest eigenstate, which
in a singlet-fission system is dominated by TT character — its small Frenkel
admixture gives the (weak) low-temperature TT-character emission. At higher T,
population leaks into Frenkel-character eigenstates and the bright vibronic
progression emerges. The physics of "which states emit" is entirely carried
by the dipole-squared matrix elements coming from EMI_OSC, not by any
hand-curated selection.

For each sideband s = 0, 1, ..., vibmax the photon energy is
    E_photon = E_ex_s + evalue[alpha] - s * hbar*omega_vib (in eV)
where hbar*omega_vib = vib_freq / eV converts the vibrational quantum into eV.
"""

import json
import numpy as np
import time as tm

from Emission_OSC import EMI_OSC


with open('parameters.json') as _f:
    _params = json.load(_f)

_E_EX_S         = _params['Energy_setting']['E_ex_s']
_VIB_FREQ       = _params['Energy_unit_exchange']['vib_freq']
_EV             = _params['Energy_unit_exchange']['eV']
_VIBMAX         = _params['geometry_parameters']['vibmax']
_STEP           = _params['Abs_plotting']['step']
_STEP_WIDTH     = _params['Abs_plotting']['step_width']
_NORMALIZED     = _params['Abs_plotting']['Normalized']
_GAMMA          = _params['Emi_plotting']['emission_gamma']
_FREQ_FAC       = _params['Emi_plotting']['emi_freq_fac_switch']
_TEMP_DEPEN     = _params['Emi_plotting']['Emi_Temp_depen']
_INITIAL_TEMP   = _params['Emi_plotting']['Initial_Temp']
_TEMP_STEP      = _params['Emi_plotting']['Temp_step']
_TOT_N_TEMP     = _params['Emi_plotting']['TOT_N_Temp']
_KB             = _params['Emi_plotting']['Kb']


class EMI_SP:
    """Emission spectrum with no Kasha approximation, no spin filtering."""

    def __init__(self, kcount, evect, evalue):
        self.kcount = kcount
        self.evect  = evect
        self.evalue = evalue

        self.E_ex_s     = _E_EX_S
        self.vib_freq   = _VIB_FREQ
        self.eV         = _EV
        self.vibmax     = _VIBMAX
        self.step       = _STEP
        self.step_width = _STEP_WIDTH
        self.normalized = _NORMALIZED
        self.gamma      = _GAMMA
        self.freq_fac   = _FREQ_FAC
        self.temp_dep   = _TEMP_DEPEN
        self.T_init     = _INITIAL_TEMP
        self.T_step     = _TEMP_STEP
        self.n_temp     = _TOT_N_TEMP if _TEMP_DEPEN else 1
        self.Kb         = _KB

        # Vibrational quantum in eV (used to shift sidebands)
        self.hw_eV = self.vib_freq / self.eV

    # ------------------------------------------------------------------
    # Populations
    # ------------------------------------------------------------------

    def populations(self, T):
        """Boltzmann populations over all eigenstates.

        At T == 0, all population sits in the absolute lowest eigenstate
        (whatever its character — typically TT in a singlet-fission system).
        At T > 0, standard Boltzmann distribution with the minimum eigenvalue
        as the reference energy.
        """
        if T <= 0.0:
            P = np.zeros(self.kcount, dtype=float)
            P[np.argmin(self.evalue)] = 1.0
            return P
        dE = self.evalue - self.evalue.min()
        boltz = np.exp(-dE / (self.Kb * T))
        return boltz / boltz.sum()

    # ------------------------------------------------------------------
    # Spectrum
    # ------------------------------------------------------------------

    def cal_EMI(self):
        # 1. Get sideband-resolved oscillator strengths
        t0 = tm.time()
        emi = EMI_OSC(self.kcount, self.evect, self.evalue)
        osc_x, osc_y = emi.gen_EMI_OSC()                       # (kcount, vibmax+1)
        osc_total = osc_x + osc_y
        print(f'EMI_OSC computed in {tm.time() - t0:.3f} s')

        # 2. Photon-energy axis
        x = self.E_ex_s + self.step_width * (np.arange(self.step) - self.step / 2)

        # 3. Loop over temperatures, fully vectorized inside each T
        t0 = tm.time()
        if self.temp_dep and self.n_temp > 1:
            emi_total = np.zeros((self.step, self.n_temp), dtype=float)
            for k in range(self.n_temp):
                T = self.T_init + k * self.T_step
                emi_total[:, k] = self._spectrum_at_T(x, osc_total, T)
        else:
            T = self.T_init
            emi_total = self._spectrum_at_T(x, osc_total, T)
        print(f'Emission spectrum computed in {tm.time() - t0:.3f} s')

        # 4. Optional normalization to unit peak
        if self.normalized:
            mx = emi_total.max()
            if mx > 0:
                emi_total = emi_total / mx

        # 5. Save
        if emi_total.ndim == 1:
            np.savetxt('MY_EMI.dat',
                       np.column_stack([x, emi_total]),
                       fmt='%.6f', delimiter='\t')
        else:
            np.savetxt('MY_EMI.dat',
                       np.column_stack([x, emi_total]),
                       fmt='%.6f', delimiter='\t')

        return x, emi_total

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spectrum_at_T(self, x, osc_total, T):
        """Compute emission spectrum at a single temperature.

        Parameters
        ----------
        x : (step,) ndarray
            Photon energy axis in eV.
        osc_total : (kcount, vibmax+1) ndarray
            |mu_X|^2 + |mu_Y|^2 per (eigenstate, sideband).
        T : float
            Temperature in Kelvin.

        Returns
        -------
        spectrum : (step,) ndarray
        """
        P = self.populations(T)                                # (kcount,)
        spectrum = np.zeros(self.step, dtype=float)

        # For each sideband s, the photon energy is E_alpha - s*hw + E_ex_s.
        # Centers: (kcount, vibmax+1) array of peak positions
        s_arr = np.arange(self.vibmax + 1)                     # (vibmax+1,)
        # centers[alpha, s] = E_ex_s + evalue[alpha] - s * hw
        centers = self.E_ex_s + self.evalue[:, None] - s_arr[None, :] * self.hw_eV

        # ω^3 factor per (alpha, s) — uses the photon energy at peak center
        if self.freq_fac:
            freq3 = centers ** 3
        else:
            freq3 = np.ones_like(centers)

        # Weighted oscillator strength per (alpha, s)
        weighted = (P[:, None] * freq3 * osc_total)            # (kcount, vibmax+1)

        # Gaussian broadening: vectorized over x and (alpha, s).
        # For each (alpha, s), add weighted[alpha, s] * exp(-((x - centers[alpha, s])/gamma)^2)
        # The full outer product (step, kcount, vibmax+1) can be memory-heavy
        # for large kcount, so loop over sidebands (small, vibmax+1).
        for s in range(self.vibmax + 1):
            mu = centers[:, s]                                 # (kcount,)
            w  = weighted[:, s]                                # (kcount,)
            # arg shape (step, kcount), then sum over kcount
            arg = (x[:, None] - mu[None, :]) / self.gamma
            spectrum += (w[None, :] * np.exp(-arg ** 2)).sum(axis=1)

        return spectrum