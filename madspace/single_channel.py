""" Implement ps-mappings
    for two dominant WWW and
    VBS diagrams
"""


from typing import Tuple, Optional
import torch
from torch import Tensor, sqrt, log, atan2

from .base import PhaseSpaceMapping, TensorList
from .helper import boost_beam, edot, lsquare
from .twoparticle import (
    tInvariantTwoParticleCOM,
    tInvariantTwoParticleLAB,
    TwoParticleLAB,
)
from .luminosity import Luminosity
from .invariants import BreitWignerInvariantBlock, UniformInvariantBlock


class SingleChannelWWW(PhaseSpaceMapping):
    """
    Dominant channel for triple WWW:

                              (k1)
                               W+
                               <
                               <
                         s12   <
    (p1) d~ --- < ---*vvvvvvvvv*vvvvvvvvv W- (k2)
                     |
                     ^ t1
                     |
    (p2) u  --- > ---*vvvvvvvvv W+ (k3)

    s-Invariants:
      shat = (k1 + k2 + k3)^2 = (p1 + p2)^2
       s12 = (k1 + k2)^2      = (k12)^2

    t-Invariants:
        t1 = (k12 - p1)^2 = (k3 - p2)^2

    Fields:
        incoming: u d~
        outgoing: W+ W- W+
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
        mz: Tensor,
        wz: Tensor,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
            mz (Tensor): mass of Z Boson
            wz (Tensor): width of Z Boson
        """
        dims_in = [(7,)]
        dims_out = [(5, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw
        self.mz = mz

        # get minimum cuts
        self.s_lab = s_lab
        self.s_hat_min = (3 * mw) ** 2

        # Define mappings
        self.luminosity = Luminosity(s_lab, self.s_hat_min)  # 2dof
        self.t1 = tInvariantTwoParticleCOM(nu=1.4)  # 2dof
        self.s12 = BreitWignerInvariantBlock(mz, wz)  # 1 dof
        self.decay = TwoParticleLAB(mw, mw)  # 2 dof

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to moment

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,7)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,5,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        r = inputs[1]
        r_lumi = r[:, :2]
        r_t1 = r[:, 2:4]
        r_s12 = r[:, 4:5]
        r_d = r[:, 5:]

        # Do luminosity and get s_hat and rapidity
        (x1x2,), det_lumi = self.luminosity.map([r_lumi])
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # Sample k12 propagator
        s12_min = (2 * self.mw) ** 2 * torch.ones_like(s_hat)
        s12_max = (sqrt(s_hat) - self.mw) ** 2
        (s12,), det_s12 = self.s12.map([r_s12], condition=[s12_min, s12_max])

        # construct initial state momenta
        p1 = torch.zeros((r_t1.shape[0], 1, 4))
        p2 = torch.zeros((r_t1.shape[0], 1, 4))
        p1[..., 0] = sqrt(s_hat) / 2
        p1[..., 3] = sqrt(s_hat) / 2
        p2[..., 0] = sqrt(s_hat) / 2
        p2[..., 3] = -sqrt(s_hat) / 2
        p_t1_in = torch.cat([p1, p2], dim=1)

        # get masses/virtualities
        m12 = sqrt(s12)
        m3 = torch.ones_like(m12) * self.mw
        m_out = torch.cat([m12, m3], dim=1)

        (p_t1_out, _, _), det_t1 = self.t1.map([r_t1, p_t1_in, m_out])

        # prepare decay of z-boson
        p_d_in = p_t1_out[:, 0]
        (p_d_out,), det_decay = self.decay.map([r_d, p_d_in])

        # Pack all momenta including initial state
        p3 = p_t1_out[:, 1:2]
        p_ext = torch.cat([p_t1_in, p_d_out, p3], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = boost_beam(p_ext, rap)
        ps_weight = det_lumi * det_s12 * det_t1 * det_decay

        return (p_ext_lab, x1x2), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        p_ext_lab = inputs[0]
        x1x2 = inputs[1]

        # Undo boosts etc
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        p_ext = boost_beam(p_ext_lab, rap, inverse=True)

        # Undo decay
        p_d_out = p_ext[:, 2:4]
        (r_d, p_d_in), det_decay_inv = self.decay.map_inverse([p_d_out])

        # Undo t-channel 2->2
        p_t1_out = torch.stack([p_d_in, p_ext[:, 4]])
        p1_2 = torch.zeros_like(x1x2[:, 0])
        p2_2 = torch.zeros_like(x1x2[:, 0])
        m2_in = torch.stack([p1_2, p2_2], dim=1)
        phi1 = torch.zeros_like(x1x2[:, 0])
        costheta1 = torch.ones_like(x1x2[:, 0])
        angles_in = torch.stack([phi1, costheta1], dim=1)
        (r_t1, _, m_out), det_t1_inv = self.t1.map_inverse([p_t1_out, m2_in, angles_in])

        # Undo s-channel sampling
        s12 = m_out[:, 0] ** 2
        s12_min = torch.zeros_like(s12)
        s12_max = (sqrt(s_hat) - self.mw) ** 2
        (r_s12,), det_s12_inv = self.s1.map([s12], condition=[s12_min, s12_max])

        # Undo lumi param
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # Pack all together
        r = torch.cat([r_lumi, r_t1, r_s12, r_d])
        r_weight = det_lumi_inv * det_s12_inv * det_t1_inv * det_decay_inv

        return (r,), r_weight

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            _, det = self.map_inverse(inputs)
            return det

        _, det = self.map(inputs)
        return det


class SingleChannelVBS(PhaseSpaceMapping):
    """
    Dominant channel for VBS :
                                      _ _ _ _ _ _
    (p1) c --- > ---*vvvvvvvvv W+ (k1)  |       |
                    |                   |       |
                    | t3                | k12   |
                    |                   |       |
                    *--- > --- s (k2)_ _|       | k123
                    g                           |
                    g t2                        |
                    g                           |
                    *--- > --- d (k3)_ _ _ _ _ _|
                    |
                    | t1
                    |
    (p2) u --- > ---*vvvvvvvvv W+ (k4)

    s-Invariants:
      shat = (k1 + k2 + k3 + k4)^2 = (p1 + p2)^2
      s123 = (k1 + k2 + k3)^2      = (k123)^2
       s12 = (k1 + k2)^2           = (k12)^2

    t-Invariants:
        t1 = (k123 - p1)^2 = (k4 - p2)^2
        t2 = (k12  - p1)^2 = (k3 + k4 - p2)^2
        t3 = (k1   - p1)^2 = (k2 + k3 + k4 - p2)^2

    Fields:
        incoming: u c
        outgoing: W+ W+ d s
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
        """
        dims_in = [(10,)]
        dims_out = [(6, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw

        # get minimum cuts
        self.s_lab = s_lab
        s_hat_min = (2 * mw) ** 2

        # Define mappings
        self.luminosity = Luminosity(s_lab, s_hat_min)  # 2dof
        self.t1 = tInvariantTwoParticleCOM(nu=1.4)  # 2dof
        self.t2 = tInvariantTwoParticleLAB(nu=1.4)  # 2dof
        self.t3 = tInvariantTwoParticleLAB(nu=1.4)  # 2dof

        self.s12 = UniformInvariantBlock()  # 1 dof
        self.s123 = UniformInvariantBlock()  # 1 dof

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to moment

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,7)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,5,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        r = inputs[1]
        r_lumi = r[:, :2]
        r_s12 = r[:, 2:3]
        r_s123 = r[:, 3:4]
        r_t1 = r[:, 4:6]
        r_t2 = r[:, 6:8]
        r_t3 = r[:, 8:10]

        # Do luminosity and get s_hat and rapidity
        (x1x2,), det_lumi = self.luminosity.map([r_lumi])
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # construct initial state momenta
        p1 = torch.zeros((r_t1.shape[0], 1, 4))
        p2 = torch.zeros((r_t1.shape[0], 1, 4))
        p1[..., 0] = sqrt(s_hat) / 2
        p1[..., 3] = sqrt(s_hat) / 2
        p2[..., 0] = sqrt(s_hat) / 2
        p2[..., 3] = -sqrt(s_hat) / 2
        p_in = torch.cat([p1, p2], dim=1)

        # Sample s-invariants
        s12_min = torch.ones_like(rap) * self.mw**2
        s12_max = (sqrt(s_hat) - self.mw) ** 2
        (s12,), det_s12 = self.s12.map([r_s12], condition=[s12_min, s12_max])

        s123_min = s12
        s123_max = (sqrt(s_hat) - self.mw) ** 2
        (s123,), det_s123 = self.s123.map([r_s123], condition=[s123_min, s123_max])

        # Then do t-invariant maps
        # get masses/virtualities
        m123 = sqrt(s123)
        m4 = torch.ones_like(m123) * self.mw
        m_t1 = torch.cat([m123, m4], dim=1)
        (k_t1, _, _), det_t1 = self.t1.map([r_t1, p_in, m_t1])

        # Then do the next
        qt2 = p_in[:, 1] - k_t1[:, 1]
        p_t2_in = torch.stack([p_in[:, 1], qt2], dim=1)
        m12 = sqrt(s12)
        m3 = torch.zeros_like(m12)
        m_t2 = torch.cat([m12, m3], dim=1)
        (k_t2, _, _), det_t2 = self.t2.map([r_t2, p_t2_in, m_t2])

        # Then do the next
        qt3 = qt2 - k_t2[:, 1]
        p_t3_in = torch.stack([p_in[:, 1], qt3], dim=1)
        m1 = torch.ones_like(m12) * self.mw
        m2 = torch.zeros_like(m12)
        m_t3 = torch.cat([m1, m2], dim=1)
        (k_t3, _, _), det_t3 = self.t2.map([r_t3, p_t3_in, m_t3])

        # Get all outgoing momenta
        k_out = torch.stack([k_t3[:, 0], k_t1[:, 1], k_t2[:, 1], k_t1[:, 1]], dim=1)

        # And then all external momenta
        p_ext = torch.cat([p_in, k_out], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = boost_beam(p_ext, rap)
        ps_weight = det_lumi * det_s12 * det_s123 * det_t1 * det_t2 * det_t3

        return (p_ext_lab, x1x2), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        pass

        # return (r,), r_weight

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            _, det = self.map_inverse(inputs)
            return det

        _, det = self.map(inputs)
        return det
