""" Implement two-particle mappings.
    Bases on the mappings described in
    https://freidok.uni-freiburg.de/data/154629"""


from typing import Tuple, Optional
import torch
import math as m
from torch import Tensor, sqrt, log, atan2, atan, tan

from .base import PhaseSpaceMapping, TensorList
from .helper import kaellen, rotate_zy, boost, lsquare, edot, pi

TF_PI = torch.tensor(pi)


def two_particle_density(s: Tensor, m1: Tensor, m2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.8) - (C.10) in [1]

    Args:s
        s (Tensor): squared COM energy of the proces with shape=(b,)
        m1 (Tensor): Mass of decay particle 1 with shape=()
        m2 (Tensor): Mass of decay particle 2 with shape=()

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    g = (2 * s) / sqrt(kaellen(s, m1**2, m2**2)) / TF_PI
    return g


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

    def _map(self, inputs: TensorList, condition: TensorList):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, p0]
                r: random numbers with shape=(b,2)
                s: squared COM energy with shape=(b,)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            logdet (Tensor): log det of mapping with shape=(b,)
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

        # get the log determinant and decay momenta
        logdet = log(two_particle_density(s, self.m1, self.m2))
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), -logdet

    def _map_inverse(self, inputs: TensorList, condition: TensorList):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
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

        # get the log determinant
        logdet = log(two_particle_density(s, self.m1, self.m2))

        return (r, s), logdet

    def _log_det(
        self,
        inputs: TensorList,
        condition: TensorList,
        inverse: bool = False,
    ):
        del condition
        s = inputs[1] if inverse else lsquare(inputs[0].sum(dim=1))
        logdet = -log(two_particle_density(s, self.m1, self.m2))
        return (-1) ** (inverse) * logdet


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
        super().__init__(dims_r=(2,), dims_p=(2, 4), dims_c=[(4,)])
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2

    def _map(self, inputs: TensorList, condition: TensorList):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, p0]
                r: random numbers with shape=(b,2)
                p0: total momentum in lab frame with shape=(b,4)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            logdet (Tensor): log det of mapping with shape=(b,)
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

        # get the log determinant and decay momenta
        logdet = log(self._density(p0))
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), -logdet

    def _map_inverse(self, inputs: TensorList, condition: TensorList):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
        """
        del condition
        p_decay = inputs[0]

        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
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

        # get the log determinant
        logdet = log(self._density(p0))

        return (r, p0), logdet

    def _log_det(
        self,
        inputs: TensorList,
        condition: TensorList,
        inverse: bool = False,
    ):
        del condition
        s = lsquare(inputs[1]) if inverse else lsquare(inputs[0].sum(dim=1))
        logdet = -log(two_particle_density(s, self.m1, self.m2))
        return (-1) ** (inverse) * logdet


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

    def _map(self, inputs: TensorList, condition: TensorList):
        raise NotImplementedError

    def _map_inverse(self, inputs: TensorList, condition: TensorList):
        raise NotImplementedError
