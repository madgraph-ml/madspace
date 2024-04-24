""" Implement three-particle mappings.
    Bases on the mappings described in
    https://freidok.uni-freiburg.de/data/154629"""


from typing import Tuple, Optional
import torch
from torch import Tensor, sqrt, log, atan2, atan, tan

from .base import PhaseSpaceMapping, TensorList
from .helper import (
    inv_rotate_zy,
    rotate_zy,
    boost,
    lsquare,
    edot,
    pi,
    three_particle_density,
)
from .invariants import UniformInvariantBlock


class ThreeParticleCOM(PhaseSpaceMapping):
    """
    Implement isotropic 3-particle phase-space, based on the mapping described in
        [1] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(self):
        dims_in = [(5,), (), (3,)]
        dims_out = [(3, 4)]
        super().__init__(dims_in, dims_out)

        self.e1_map = UniformInvariantBlock()
        self.e2_map = UniformInvariantBlock()

    def _map(self, inputs: TensorList, condition: TensorList):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, s, m_out]
                r: random numbers with shape=(b,5)
                s: squared COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)

        Returns:
            p_decay (Tensor): decay momenta (COM frame) with shape=(b,3,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, s, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        p3 = torch.zeros(r.shape[0], 4, device=r.device)
        with torch.no_grad():
            if torch.any(s < 0):
                raise ValueError(f"s needs to be always positive")

        # Get virutalities/masses
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (p10,), p1_det = self.e1_map.map([r[:, 0]], [m1, E1a])

        # Define the energy p10
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (p20,), p2_det = self.e1_map.map([r[:, 1]], [E2a, E2b])

        # Define angles
        phi = 2 * pi * r[:, 2]
        costheta = 2 * r[:, 3] - 1
        beta = 2 * pi * r[:, 4]
        det_omega = 8 * pi**2

        # calculate cosalpha
        num_alpha_1 = 2 * sqrt(s) * (sqrt(s) / 2 - p10 - p20)
        num_alpha_2 = m1sq + m2sq + 2 * p10 * p20 - m3sq
        denom_alpha = 2 * sqrt(p10**2 - m1sq) * sqrt(p20**2 - m2sq)
        cosalpha = (num_alpha_1 + num_alpha_2) / denom_alpha

        # Fill momenta
        p1[:, 0] = p10
        p1[:, 3] = sqrt(p10**2 - m1sq)
        p2[:, 0] = p20
        p2[:, 3] = sqrt(p20**2 - m2sq)

        # Do rotations
        p1 = rotate_zy(p1, phi, costheta)
        # Double rotate p2
        p2 = rotate_zy(p2, beta, cosalpha)
        p2 = rotate_zy(p2, phi, costheta)

        # Get final momentum
        p3[:, 0] = sqrt(s) - p10 - p20
        p3[:, 1:] = -p1[:, 1:] - p2[:, 1:]

        # get the density and decay momenta
        # (C.10) in [2] == (C.8)/(4PI)
        gs = three_particle_density() * p1_det * p2_det * det_omega
        p_decay = torch.stack([p1, p2, p3], dim=1)

        return (p_decay,), gs

    def _map_inverse(self, inputs: TensorList, condition: TensorList):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (COM frame) with shape=(b,3,4)

        Returns:
            r (Tensor): random numbers with shape=(b,5)
            s (Tensor): squared COM energy with shape=(b,)
            m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]
        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        m_out = sqrt(lsquare(p_decay))

        # particle features
        p1 = p_decay[:, 0]
        p2 = p_decay[:, 1]
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        p10 = p1[:, 0]
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (r_p1,), p1_det_inv = self.e1_map.map([p10], [m1, E1a])

        # Define the energy p10
        p20 = p2[:, 0]
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (r_p2,), p2_det_inv = self.e1_map.map([p20], [E2a, E2b])

        # get p1/2 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))
        p2mag = sqrt(edot(p2[:, 1:], p2[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r_phi = phi / 2 / pi
        r_theta = (costheta + 1) / 2

        # Get last angle
        p2 = inv_rotate_zy(p2, phi, costheta)
        beta = atan2(p2[:, 2], p2[:, 1])
        r_beta = beta / 2 / pi
        det_omega_inv = 1 / (8 * pi**2)

        # Pack all together and get full density
        gs = p1_det_inv * p2_det_inv * det_omega_inv / three_particle_density()
        r = torch.stack([r_p1, r_p2, r_phi, r_theta, r_beta], dim=1)

        return (r, s, m_out), gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        """Returns the density only of the mapping"""
        if inverse:
            _, density = self.map_inverse(inputs, condition)
            return density
        _, density = self.map(inputs, condition)
        return density


class ThreeParticleLAB(PhaseSpaceMapping):
    """
    Implement isotropic 3-particle phase-space, based on the mapping described in
        [1] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(self):
        dims_in = [(5,), (4,), (3,)]
        dims_out = [(3, 4), (2,)]
        super().__init__(dims_in, dims_out)

        self.e1_map = UniformInvariantBlock()
        self.e2_map = UniformInvariantBlock()

    def _map(self, inputs: TensorList, condition: TensorList):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, p0, m_out]
                r: random numbers with shape=(b,5)
                p0: incoming momentum in lab frame with shape=(b,4)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)

        Returns:
            p_lab (Tensor): decay momenta (lab frame) with shape=(b,3,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, p0, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        p3 = torch.zeros(r.shape[0], 4, device=r.device)
        s = lsquare(p0)

        with torch.no_grad():
            if torch.any(s < 0):
                raise ValueError(f"s needs to be always positive")

        # Get virutalities/masses
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (p10,), p1_det = self.e1_map.map([r[:, 0]], [m1, E1a])

        # Define the energy p10
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (p20,), p2_det = self.e1_map.map([r[:, 1]], [E2a, E2b])

        # Define angles
        phi = 2 * pi * r[:, 2]
        costheta = 2 * r[:, 3] - 1
        beta = 2 * pi * r[:, 4]
        det_omega = 8 * pi**2

        # calculate cosalpha
        num_alpha_1 = 2 * sqrt(s) * (sqrt(s) / 2 - p10 - p20)
        num_alpha_2 = m1sq + m2sq + 2 * p10 * p20 - m3sq
        denom_alpha = 2 * sqrt(p10**2 - m1sq) * sqrt(p20**2 - m2sq)
        cosalpha = (num_alpha_1 + num_alpha_2) / denom_alpha

        # Fill momenta
        p1[:, 0] = p10
        p1[:, 3] = sqrt(p10**2 - m1sq)
        p2[:, 0] = p20
        p2[:, 3] = sqrt(p20**2 - m2sq)

        # Do rotations
        p1 = rotate_zy(p1, phi, costheta)
        # Double rotate p2
        p2 = rotate_zy(p2, beta, cosalpha)
        p2 = rotate_zy(p2, phi, costheta)

        # Get final momentum
        p3[:, 0] = sqrt(s) - p10 - p20
        p3[:, 1:] = -p1[:, 1:] - p2[:, 1:]

        # boost into lab-frame
        p_decay = torch.stack([p1, p2, p3], dim=1)
        p_lab = boost(p_decay, p0[:, None])

        # get the full density
        gs = three_particle_density() * p1_det * p2_det * det_omega

        return (p_lab,), gs

    def _map_inverse(self, inputs: TensorList, condition: TensorList):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_lab: decay momenta (lab frame) with shape=(b,3,4)

        Returns:
            r (Tensor): random numbers with shape=(b,5)
            p0 (Tensor): incoming momentum in lab frame with shape=(b,4)
            m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_lab = inputs[0]

        # Decaying particle in lab-frame
        p0 = p_lab.sum(dim=1, keepdim=True)
        s = lsquare(p0)
        m_out = sqrt(lsquare(p_lab))

        # boost into COM-frame
        p_decay = boost(p_lab, p0, inverse=True)

        # particle features
        p1 = p_decay[:, 0]
        p2 = p_decay[:, 1]
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        p10 = p1[:, 0]
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (r_p1,), p1_det_inv = self.e1_map.map([p10], [m1, E1a])

        # Define the energy p10
        p20 = p2[:, 0]
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (r_p2,), p2_det_inv = self.e1_map.map([p20], [E2a, E2b])

        # get p1/2 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))
        p2mag = sqrt(edot(p2[:, 1:], p2[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r_phi = phi / 2 / pi
        r_theta = (costheta + 1) / 2

        # Get last angle
        p2 = inv_rotate_zy(p2, phi, costheta)
        beta = atan2(p2[:, 2], p2[:, 1])
        r_beta = beta / 2 / pi
        det_omega_inv = 1 / (8 * pi**2)

        # Pack all together and get full density
        gs = p1_det_inv * p2_det_inv * det_omega_inv / three_particle_density()
        r = torch.stack([r_p1, r_p2, r_phi, r_theta, r_beta], dim=1)

        return (r, s, m_out), gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        """Returns the density only of the mapping"""
        if inverse:
            _, density = self.map_inverse(inputs, condition)
            return density
        _, density = self.map(inputs, condition)
        return density
