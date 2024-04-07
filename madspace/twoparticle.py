""" Implement two-particle mappings.
    Bases on the mappings described in
    [1] https://freidok.uni-freiburg.de/data/154629
    [2] https://arxiv.org/abs/hep-ph/0008033
"""


from typing import Tuple, Optional
import torch
from torch import Tensor, sqrt, log, atan2

from .base import PhaseSpaceMapping, TensorList
from .helper import two_particle_density, kaellen, rotate_zy, boost, lsquare, edot, pi

TF_PI = torch.tensor(pi)


class TwoParticleCOM(PhaseSpaceMapping):
    """
    Implement isotropic 2-particle phase-space, based on the mapping described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
    ):
        """
        Args:
            m1 (Tensor): Mass of decay particle 1
            m2 (Tensor): Mass of decay particle 2
        """
        super().__init__(dims_in=[(2,), ()], dims_out=[(2, 4)], dims_c=None)
        self.m1 = m1
        self.m2 = m2

    def map(self, inputs, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of two tensors [r, s]
                r: random numbers with shape=(b,2)
                s: squared COM energy with shape=(b,)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        r, s = inputs[0], inputs[1]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        if torch.any(s < 0):
            raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * TF_PI * r1
        costheta = 2 * r2 - 1

        # Define the momenta (in COM frame of decaying particle)
        p1[:, 0] = (s + self.m1**2 - self.m2**2) / (2 * sqrt(s))
        p2[:, 0] = (s + self.m2**2 - self.m1**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, self.m1**2, self.m2**2)) / (2 * sqrt(s))

        # Rotate and define p2 spatial components
        p1 = rotate_zy(p1, phi, costheta)
        p2[:, 1:] = -p1[:, 1:]

        # get the density and decay momenta
        # (C.10) in [2] == (C.8)/(4PI)
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), 1 / gs

    def map_inverse(self, inputs, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]
        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        p1 = p_decay[:, 0]

        # get p1 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r1 = phi / 2 / TF_PI
        r2 = (costheta + 1) / 2
        r = torch.stack([r1, r2], dim=1)

        # get the density and decay momenta
        # (C.10) in [2] == (C.8)/(4PI)
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)

        return (r, s), gs

    def density(self, inputs, condition=False, inverse=False):
        del condition
        if inverse:
            s = inputs[1]
            gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
            return gs

        s = lsquare(inputs[0].sum(dim=1))
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
        return 1 / gs


class TwoParticleLAB(PhaseSpaceMapping):
    """
    Implement isotropic 2-particle phase-space, based on the mapping described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
    ):
        super().__init__(dims_in=[(2,), (4,)], dims_out=[(2, 4)], dims_c=None)
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2

    def map(self, inputs, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, p0]
                r: random numbers with shape=(b,2)
                p0: total momentum in lab frame with shape=(b,4)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, p0 = inputs[0], inputs[1]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        s = lsquare(p0)
        if torch.any(s < 0):
            raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * TF_PI * r1
        costheta = 2 * r2 - 1

        # Define the momenta (in COM frame of decaying particle)
        p1[:, 0] = (s + self.m1**2 - self.m2**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, self.m1**2, self.m2**2)) / (2 * sqrt(s))

        # First rotate, then boost into lab-frame
        p1 = rotate_zy(p1, phi, costheta)
        p1 = boost(p1, p0)
        p2 = p0 - p1

        # get the density and decay momenta
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), 1 / gs

    def map_inverse(self, inputs, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]

        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        p1 = p_decay[:, 0]

        # Boost p1 into COM
        p1 = boost(p1, -p0)

        # get p1 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r1 = phi / 2 / TF_PI
        r2 = (costheta + 1) / 2
        r = torch.stack([r1, r2], dim=1)

        # get the density
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)

        return (r, p0), gs

    def density(self, inputs, condition=False, inverse=False):
        del condition
        if inverse:
            s = lsquare(inputs[1])
            gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
            return gs

        s = lsquare(inputs[0].sum(dim=1))
        gs = two_particle_density(s, self.m1, self.m2) / (4 * TF_PI)
        return 1 / gs


class tChannelTwoParticle(PhaseSpaceMapping):
    """
    Implement anisotropic decay, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
        [2] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
        mt: Optional[Tensor] = None,
        wt: Optional[Tensor] = None,
    ):
        super().__init__(dims_r=(2,), dims_p=(2, 4), dims_c=(2, 4))
        self.mt = mt  # Mass of t-channel particle
        self.wt = wt  # Width of t-channel particle
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2

        # # TODO make limits smin and smax inputs of mapping and not fixedß
        # if mt is None:
        #     self.t_map = sChannelMasslessBlock(nu=2.0)
        # elif wt is None:
        #     self.t_map = sChannelZeroWidthBlock(nu=2.0, mass=mt)
        # else:
        #     self.t_map = sChannelBWBlock(mass=mt, width=wt)

    def map(self, inputs, condition=None):
        raise NotImplementedError

    def map_inverse(self, inputs, condition=None):
        raise NotImplementedError
    
    def density(self, inputs, condition= None, inverse=False):
        return NotImplementedError
