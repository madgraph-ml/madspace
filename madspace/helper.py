""" Helper functions needed for phase-space mappings """

import torch
from torch import Tensor, cos, sin, cosh, sinh, sqrt, log
from math import pi


MINKOWSKI = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))

def two_particle_density(s: Tensor, m1: Tensor, m2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.8) in [1]

    Args:
        s (Tensor): squared COM energy of the proces with shape=(b,)
        m1 (Tensor): Mass of decay particle 1 with shape=(b,)
        m2 (Tensor): Mass of decay particle 2 with shape=(b,)

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    g = (8 * s) / sqrt(kaellen(s, m1**2, m2**2))
    return g


def tinv_two_particle_density(s: Tensor, p1_2: Tensor, p2_2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.22) in [1]

    Args:
        s (Tensor): squared COM energy of the proces with shape=(b,)
        p1_2 (Tensor): Virtuality of incoming particle 1 with shape=(b,)
        p2_2 (Tensor): Virtuality of incoming particle 2 with shape=(b,)

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    g = 4 * sqrt(kaellen(s, p1_2, p2_2))
    return g


def kaellen(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Definition of the standard kaellen function [1]

    [1] https://en.wikipedia.org/wiki/Källén_function

    Args:
        a (Tensor): input 1
        b (Tensor): input 2
        c (Tensor): input 3

    Returns:
        Tensor: Kaellen function
    """
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a


def rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs rotation around y- and z-axis:

        p -> p' = R_z(phi).R_y(theta).p

    with the explizit matrice following the conventions in [1]
    to achieve proper spherical coordinates [2]:

    R_z = (  1       0         0      0  )
          (  0   cos(phi)  -sin(phi)  0  )
          (  0   sin(phi)   cos(phi)  0  )
          (  0       0         0      1  )

    R_y = (  1       0       0      0       )
          (  0   cos(theta)  0  sin(theta)  )
          (  0       0       1      0       )
          (  0  -sin(theta)  0  cos(theta)  )

    For a 3D vector v = (0, 0, |v|)^T this results in the general spherical
    coordinate vector

        v -> v' = (  |v|*sin(theta)*cos(phi)  )
                  (  |v|*sin(theta)*sin(phi)  )
                  (  |v|*cos(theta)           )

    [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    [2] https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Args:
        p (Tensor): 4-momentum to rotate with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)

    # Define the rotation
    q0 = p[..., 0]
    q1 = (
        p[..., 1] * costheta * cos(phi)
        + p[..., 3] * sintheta * cos(phi)
        - p[..., 2] * sin(phi)
    )
    q2 = (
        p[..., 1] * costheta * sin(phi)
        + p[..., 3] * sintheta * sin(phi)
        + p[..., 2] * cos(phi)
    )
    q3 = p[..., 3] * costheta - p[:, 1] * sintheta

    return torch.stack((q0, q1, q2, q3), dim=-1)


def inv_rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs inverse rotation around y- and z-axis:

        p' -> p = R_y(-theta).R_z(-phi).p

    Args:
        p (Tensor): rotated 4-momentum inverse with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)

    # Define the rotation
    q0 = p[..., 0]
    q1 = (
        p[..., 1] * costheta * cos(phi)
        + p[..., 2] * costheta * sin(phi)
        - p[..., 3] * sintheta
    )
    q2 = p[..., 2] * cos(phi) - p[:, 1] * sin(phi)
    q3 = (
        p[..., 3] * costheta
        + p[..., 1] * sintheta * cos(phi)
        + p[..., 2] * sintheta * sin(phi)
    )

    return torch.stack((q0, q1, q2, q3), dim=-1)


def lsquare(a: Tensor) -> Tensor:
    """Gives the lorentz invariant a^2 using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    return torch.einsum("...d,dd,...d->...", a, MINKOWSKI, a)


def ldot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the Lorentz inner product ab using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)
        b (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    return torch.einsum("...d,dd,...d->...", a, MINKOWSKI, b)


def edot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the euclidean inner product ab using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)
        b (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, b)


def boost(k: Tensor, p_boost: Tensor) -> Tensor:
    """
    Boost k into the frame of p_boost in argument.
    This means that the following command, for any vector k=(E, px, py, pz)
    gives:

        k  -> k' = boost(k, -k) = (M,0,0,0)
        k' -> k  = boost(k', k) = (E, px, py, pz)

    Args:
        k (Tensor): input vector with shape=(b,n,4)/(b,4)
        p_boost (Tensor): boosting vector with shape=(b,1,4)/(b,4)

    Returns:
        k' (Tensor): boosted vector with shape=(b,n,4)/(b,4)
    """
    # Make sure energy is > 0 even after momentum flip (for inverse boost)
    p_boost[..., 0] = torch.abs(p_boost[..., 0])

    # Perform the boost
    rsq = sqrt(lsquare(p_boost))
    k0 = edot(k, p_boost) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + c1 * p_boost[..., 1]
    k2 = k[..., 2] + c1 * p_boost[..., 2]
    k3 = k[..., 3] + c1 * p_boost[..., 3]

    return torch.stack((k0, k1, k2, k3), dim=-1)


def boost_beam(
    q: Tensor,
    rapidity: Tensor,
    inverse: bool = False,
) -> Tensor:
    """Boosts q along the beam axis with given rapidity

    Args:
        q (Tensor): input vector with shape=(b,n,4)/(b,4)
        rapidity (Tensor): boosting parameter with shape=(b,1,1)/(b,1)
        inverse (bool, optional): inverse boost. Defaults to False.

    Returns:
        q' (Tensor): boosted vector with shape=(b,n,4)
    """
    sign = -1.0 if inverse else 1.0

    pi0 = q[..., 0] * cosh(rapidity) + sign * q[..., 3] * sinh(rapidity)
    pix = q[..., 1]
    piy = q[..., 2]
    piz = q[..., 3] * cosh(rapidity) + sign * q[..., 0] * sinh(rapidity)

    return torch.stack((pi0, pix, piy, piz), dim=-1)


def map_fourvector_rambo(r: Tensor) -> Tensor:
    """Transform unit hypercube points into into four-vectors.

    Args:
        r (Tensor): 4n random numbers with shape=(b,n,4)

    Returns:
        q (Tensor): n 4-Momenta with shape=(b,n,4)
    """
    costheta = 2.0 * r[:, :, 0] - 1.0
    phi = 2.0 * pi * r[:, :, 1]

    q = torch.zeros_like(r)
    q[:, :, 0] = -log(r[:, :, 2] * r[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * sqrt(1 - costheta**2) * cos(phi)
    q[:, :, 2] = q[:, :, 0] * sqrt(1 - costheta**2) * sin(phi)
    q[:, :, 3] = q[:, :, 0] * costheta

    return q


def two_body_decay_factor(
    M_i_minus_1: Tensor,
    M_i: Tensor,
    m_i_minus_1: Tensor,
) -> Tensor:
    """Gives two-body decay factor from recursive n-body phase space"""
    return (
        1.0
        / (8 * M_i_minus_1**2)
        * sqrt(
            (M_i_minus_1**2 - (M_i + m_i_minus_1) ** 2)
            * (M_i_minus_1**2 - (M_i - m_i_minus_1) ** 2)
        )
    )