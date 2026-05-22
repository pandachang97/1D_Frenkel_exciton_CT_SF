"""Emission oscillator strengths for transitions from excited eigenstates to
specific vibrational configurations of the electronic ground state.

For each eigenstate alpha and each terminal vibrational sideband index s
(meaning: the ground-state manifold with s total vibrational quanta distributed
across sites in some way), this computes |<G_s | mu | alpha>|^2.

The emission ground state |G_s> with s quanta can be realized in multiple ways:
- s=0: all sites in vibrational ground state (one configuration)
- s=l1 (l1 >= 1): one site has l1 quanta, others have zero (multiple sites possible)
- s=l1+m1 (l1, m1 >= 1): two sites carry quanta (the "2P ground" sideband)
- s=l1+m1+n1: three sites (the "3P ground" sideband)

The emission to each configuration is computed and added incoherently to the
sideband index s = l1 + m1 + n1 (or 0 if all are zero). This matches the
Fortran reference.

Only 1P-block coefficients of the eigenvector enter the 0-0 line, but higher
sidebands pick up contributions from 2P/3P basis components (vibrationally hot
spectators that overlap with the ground manifold's vibrational quanta). CT, CTv,
TP, TPv basis states all have <G|mu|n> = 0 in the electric-dipole approximation
and contribute zero directly to emission (though they contribute indirectly via
their admixture into the eigenstate's 1P/2P/3P amplitudes).
"""

import json
import numpy as np

from basis_set import IBS
from FC_factor import FCF


with open('parameters.json') as _f:
    _params = json.load(_f)

_NCHROM     = _params['geometry_parameters']['Nchrom']
_VIBMAX     = _params['geometry_parameters']['vibmax']
_THETA      = _params['geometry_parameters']['theta']
_LAM_GE_S   = _params['huang_ryhs_factors']['LamGE_S']
_ADD_DOUBLE = _params['basis_set_options']['add_double']
_ADD_TRIPPLE = _params['basis_set_options']['add_tripple']
_EMI_CHECK  = _params['Emi_plotting']['emi_OSC_check']


class EMI_OSC:
    """Compute emission oscillator strengths to all vibrational sidebands.

    Parameters
    ----------
    kcount : int
        Total Hamiltonian dimension.
    evect : (kcount, kcount) ndarray
        Eigenvectors as columns.
    evalue : (kcount,) ndarray
        Eigenvalues in eV.
    """

    def __init__(self, kcount, evect, evalue):
        self.kcount = kcount
        self.evect  = evect
        self.evalue = evalue

        self.Nchrom      = _NCHROM
        self.vibmax      = _VIBMAX
        self.theta_deg   = _THETA          # never mutated
        self.lam         = _LAM_GE_S
        self.add_double  = _ADD_DOUBLE
        self.add_tripple = _ADD_TRIPPLE
        self.emi_check   = _EMI_CHECK

        # Build basis indexing in the same order as Hamiltonian.py:
        # 1P, then 2P (if enabled), then 3P (if enabled). Higher-spin /
        # CT / TP blocks live above these but are never accessed here
        # since their <G|mu|n> is zero.
        self._ibs = IBS()
        self.Index_single = self._ibs.arr_1p()
        if self.add_double:
            self.Index_double = self._ibs.arr_2p()
        if self.add_tripple:
            self.Index_tripple = self._ibs.arr_3p()

        # Cache FC factors: fc[m, n] = <m | n>_FC for the same lambda
        fcf = FCF()
        V = self.vibmax + 1
        self._fc = np.array(
            [[fcf.gen_FCF(m, n, self.lam) for n in range(V)] for m in range(V)],
            dtype=float,
        )

        # Output: (kcount, vibmax+1) — second index s = sideband (0-s emission)
        self.Emi_osci_stre_x = np.zeros((self.kcount, V), dtype=float)
        self.Emi_osci_stre_y = np.zeros((self.kcount, V), dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def gen_EMI_OSC(self):
        """Compute |mu_X|^2 and |mu_Y|^2 for every eigenstate and sideband."""
        theta_rad = np.deg2rad(self.theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        # ---- 0-0 sideband: emission to G with zero vibrations everywhere ----
        # Only 1P basis states contribute. proj_0[lab(l, l1)] = FC(0, l1).
        proj_0 = np.zeros(self.kcount, dtype=float)
        for l in range(self.Nchrom):
            for l1 in range(self.vibmax + 1):
                lab = self.Index_single[self._ibs.order_1p(l, l1)]
                proj_0[lab] = self._fc[0, l1]
        mu_0 = proj_0 @ self.evect                              # (kcount,)
        self.Emi_osci_stre_x[:, 0] = (cos_t * mu_0) ** 2
        self.Emi_osci_stre_y[:, 0] = (sin_t * mu_0) ** 2

        # ---- 0-s sidebands for s >= 1, one site carries s quanta ----
        # The terminal ground configuration has site l with s = l1 quanta,
        # all other sites in vibrational ground. Contributions come from:
        #   (a) 1P amplitude at (l, m1) with FC(l1, m1)
        #   (b) 2P amplitude at (j, j1; l, l1) with FC(0, j1)  -- 2P-1P channel
        # Each terminal configuration is one site x s value; loop over l.
        if self.vibmax > 0:
            for l in range(self.Nchrom):
                for s in range(1, self.vibmax + 1):
                    proj = np.zeros(self.kcount, dtype=float)
                    # (a) 1P part: exciton at site l, any vibrations m1
                    for m1 in range(self.vibmax + 1):
                        lab = self.Index_single[self._ibs.order_1p(l, m1)]
                        proj[lab] = self._fc[s, m1]
                    # (b) 2P part: exciton at site j != l with j1 quanta;
                    #     spectator at site l with offset = s-1 (physical s).
                    if self.add_double:
                        for j in range(self.Nchrom):
                            if j == l:
                                continue
                            for j1 in range(self.vibmax + 1):
                                # Physical constraint: exciton (j1) + spectator (s) <= vibmax
                                if j1 + s > self.vibmax:
                                    continue
                                lab = self.Index_double[
                                    self._ibs.order_2p(j, j1, l, s - 1)
                                ]
                                proj[lab] = self._fc[0, j1]
                    mu = proj @ self.evect                      # (kcount,)
                    self.Emi_osci_stre_x[:, s] += (cos_t * mu) ** 2
                    self.Emi_osci_stre_y[:, s] += (sin_t * mu) ** 2

        # ---- 0-(l1+m1) sidebands: two sites carry quanta ----
        # Terminal: site l with l1>=1, site m with m1>=1 (l < m to avoid
        # double-counting). Sideband index s = l1 + m1.
        # Contributions:
        #   (i)  2P amplitude with exciton at m, spectator at l, with FC(m1, j1)
        #   (ii) 2P amplitude with exciton at l, spectator at m, with FC(l1, j1)
        #   (iii) 3P amplitude with exciton at neighbor j of l, two spectators
        if self.add_double:
            for l in range(self.Nchrom):
                for m in range(self.Nchrom):
                    if l >= m:
                        continue
                    for l1 in range(1, self.vibmax + 1):
                        for m1 in range(1, self.vibmax + 1):
                            if l1 + m1 > self.vibmax:
                                continue
                            s = l1 + m1
                            proj = np.zeros(self.kcount, dtype=float)
                            # (i) exciton at m with j1, spectator at l with l1
                            for j1 in range(self.vibmax + 1):
                                if j1 + l1 > self.vibmax:
                                    continue
                                lab = self.Index_double[
                                    self._ibs.order_2p(m, j1, l, l1 - 1)
                                ]
                                proj[lab] += self._fc[m1, j1]
                            # (ii) exciton at l with j1, spectator at m with m1
                            for j1 in range(self.vibmax + 1):
                                if j1 + m1 > self.vibmax:
                                    continue
                                lab = self.Index_double[
                                    self._ibs.order_2p(l, j1, m, m1 - 1)
                                ]
                                proj[lab] += self._fc[l1, j1]
                            # (iii) 3P amplitude with exciton at j, neighbor of l,
                            # and two spectators at l (l1) and m (m1)
                            if self.add_tripple:
                                for j in range(self.Nchrom):
                                    if j == l or j == m:
                                        continue
                                    if abs(l - j) != 1:        # nearest-neighbor only
                                        continue
                                    for j1 in range(self.vibmax + 1):
                                        if j1 + l1 + m1 > self.vibmax:
                                            continue
                                        lab = self.Index_tripple[
                                            self._ibs.order_3p(
                                                j, j1, l, l1 - 1, m, m1 - 1
                                            )
                                        ]
                                        proj[lab] += self._fc[0, j1]
                            mu = proj @ self.evect
                            self.Emi_osci_stre_x[:, s] += (cos_t * mu) ** 2
                            self.Emi_osci_stre_y[:, s] += (sin_t * mu) ** 2

        # ---- 0-(l1+m1+n1) sidebands: three sites carry quanta ----
        # Terminal: ordered triple l < m < n with l1, m1, n1 >= 1.
        # Exciton sits at the middle site m (nearest-neighbor approx).
        if self.add_tripple:
            for l in range(self.Nchrom):
                for m in range(self.Nchrom):
                    if l >= m:
                        continue
                    for n in range(self.Nchrom):
                        if m >= n:
                            continue
                        if abs(l - m) != 1:                    # m must be neighbor of l
                            continue
                        for l1 in range(1, self.vibmax + 1):
                            for m1 in range(1, self.vibmax + 1):
                                for n1 in range(1, self.vibmax + 1):
                                    if l1 + m1 + n1 > self.vibmax:
                                        continue
                                    s = l1 + m1 + n1
                                    proj = np.zeros(self.kcount, dtype=float)
                                    for j1 in range(self.vibmax + 1):
                                        if j1 + l1 + n1 > self.vibmax:
                                            continue
                                        lab = self.Index_tripple[
                                            self._ibs.order_3p(
                                                m, j1, l, l1 - 1, n, n1 - 1
                                            )
                                        ]
                                        proj[lab] += self._fc[m1, j1]
                                    mu = proj @ self.evect
                                    self.Emi_osci_stre_x[:, s] += (cos_t * mu) ** 2
                                    self.Emi_osci_stre_y[:, s] += (sin_t * mu) ** 2

        if self.emi_check:
            np.savetxt('EMI_OSC_X.txt', self.Emi_osci_stre_x,
                       fmt='%4.6f', delimiter=' ')
            np.savetxt('EMI_OSC_Y.txt', self.Emi_osci_stre_y,
                       fmt='%4.6f', delimiter=' ')
            print('Emission oscillator strengths written to EMI_OSC_*.txt')

        return self.Emi_osci_stre_x, self.Emi_osci_stre_y