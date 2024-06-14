""" Implement propagator mappings.

    Based on the mappings described in
    [1] https://arxiv.org/abs/hep-ph/0206070v2

    and described more precisely in
    [2] https://arxiv.org/abs/hep-ph/0008033
    [3] https://freidok.uni-freiburg.de/data/154629
"""


from torch import Tensor, sqrt
import torch
from .kinematics import kaellen, EPS


def costheta_to_invt(
    s: Tensor,
    p1_2: Tensor,
    p2_2: Tensor,
    m1: Tensor,
    m2: Tensor,
    costheta: Tensor,
) -> Tensor:
    """Mandelstam invariant t=(p1-k1)^2 formula (C.21) in https://arxiv.org/pdf/hep-ph/0008033.pdf
    p=p1+p2 is at rest;
    p1, p2 are opposite along z-axis
    k1, k4 are opposite along the direction defined by theta
    theta is the angle in the COM frame between p1 & k1
    """
    num1 = (s + m1**2 - m2**2) * (s + p1_2 - p2_2)
    num2 = sqrt(kaellen(s, m1**2, m2**2)) * sqrt(kaellen(s, p1_2, p2_2)) * costheta
    num = num1 - num2
    t = m1**2 + p1_2 - num / (2 * s)
    return torch.clamp_max_(t, -EPS)


def invt_to_costheta(
    s: Tensor,
    p1_2: Tensor,
    p2_2: Tensor,
    m1: Tensor,
    m2: Tensor,
    t: Tensor,
) -> Tensor:
    """
    https://arxiv.org/pdf/hep-ph/0008033.pdf Eq.(C.21)
    invert t=(p1-k1)^2 to cos_theta = ...
    """
    num1 = (t - m1**2 - p1_2) * 2 * s
    num2 = (s + m1**2 - m2**2) * (s + p1_2 - p2_2)
    num = num1 + num2
    denom = sqrt(kaellen(s, m1**2, m2**2)) * sqrt(kaellen(s, p1_2, p2_2))
    costheta = num / denom
    return torch.clamp_(costheta, -1.0, 1.0)
