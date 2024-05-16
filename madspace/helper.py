""" Helper functions needed for phase-space mappings """

import torch
from torch import Tensor, cos, sin, cosh, sinh, sqrt, log, abs, atan2, clip
from math import pi


MINKOWSKI = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))


def two_particle_density(s: Tensor, p1_2: Tensor, p2_2: Tensor) -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (C.8) in [1]

    Args:
        s (Tensor): squared COM energy of the proces with shape=(b,)
        p1_2 (Tensor): Mass/virtualiy of outgoing particle 1 with shape=(b,)
        p2_2 (Tensor): Mass/virtualiy of outgoing particle 1 with shape=(b,)

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    # No overall (2*pi)^(-2) here!
    g2 = sqrt(kaellen(s, p1_2, p2_2)) / (8 * s)
    return g2


def three_particle_density() -> Tensor:
    """Calculates the associated phase-space density
    according to Eq. (G.12) in [2]

    Returns:
        g (Tensor): returns the density with shape=(b,)
    """
    # No overall (2*pi)^(-5) here!
    g2 = torch.tensor(1 / 8.0)
    return g2


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
    # No overall (2*pi)^(-2) here!
    g = 1 / (4 * sqrt(kaellen(s, p1_2, p2_2)))
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


def rapidity(p: Tensor) -> Tensor:
    """Gives the rapidity of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    Es = p[..., 0]
    Pz = p[..., 3]

    y = 0.5 * log((Es + Pz) / (Es - Pz))
    return torch.where(Es == 0, 99.0, y)


def phi(p: Tensor) -> Tensor:
    """Gives the azimuthal phi of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    phi = atan2(p[..., 2], p[..., 1])
    return phi


def pT2(p: Tensor) -> Tensor:
    """Gives the squared pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pt2 = p[..., 1] ** 2 + p[..., 2] ** 2
    return pt2


def pT(p: Tensor) -> Tensor:
    """Gives the pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pT2(p))


def vec3(p: Tensor) -> Tensor:
    """Gives the 3-vector of 4-mometum

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    return p[..., 1:]


def delta_rap(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: delta_y with shape=(b,...)
    """
    dy = rapidity(p) - rapidity(q)
    return torch.abs(dy)


def delta_phi(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dphi = torch.abs(phi(p) - phi(q))
    dphi = torch.where(dphi > pi, 2 * pi - dphi, dphi)
    return dphi


def deltaR(p: Tensor, q: Tensor) -> Tensor:
    """Gives DeltaR between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dy = delta_rap(p, q)
    dphi = delta_phi(p, q)
    return sqrt(dy**2 + dphi**2)


def pmag2(p: Tensor) -> Tensor:
    """Gives the squared three-momentum |p_vec|^2

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pmag2 = esquare(p[..., 1:])
    return pmag2


def pmag(p: Tensor) -> Tensor:
    """Gives the absolute three-momentum |p_vec|

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pmag2(p))


def costheta(p: Tensor) -> Tensor:
    """Gives the costheta angle of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return p[..., 3] / pmag(p)


def mass(a: Tensor) -> Tensor:
    """Gives the mass of a particle

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    #return sqrt(abs(lsquare(a)))
    return sqrt(clip(lsquare(a), min=0))


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


def esquare(a: Tensor) -> Tensor:
    """Gives the euclidean square a^2 using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Square with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, a)


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


def boost(k: Tensor, p_boost: Tensor, inverse: bool = False) -> Tensor:
    """
    Boost k into the frame of p_boost in argument.
    This means that the following command, for any vector k=(E, px, py, pz)
    gives:

        k  -> k' = boost(k, k, inverse=True) = (M,0,0,0)
        k' -> k  = boost(k', k) = (E, px, py, pz)

    Args:
        k (Tensor): input vector with shape=(b,n,4)/(b,4)
        p_boost (Tensor): boosting vector with shape=(b,1,4)/(b,4)
        inverse (bool): if boost is performed inverse or forward

    Returns:
        k' (Tensor): boosted vector with shape=(b,n,4)/(b,4)
    """
    # Change sign if inverse boost is performed
    sign = -1.0 if inverse else 1.0

    # Perform the boost
    # This is in fact a numerical more stable implementation then often used
    rsq = sqrt(lsquare(p_boost))
    k0 = (k[..., 0] * p_boost[..., 0] + sign * edot(k[..., 1:], p_boost[..., 1:])) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + sign * c1 * p_boost[..., 1]
    k2 = k[..., 2] + sign * c1 * p_boost[..., 2]
    k3 = k[..., 3] + sign * c1 * p_boost[..., 3]

    return torch.stack((k0, k1, k2, k3), dim=-1)


def boost_beam(
    q: Tensor,
    rapidity: Tensor,
    inverse: bool = False,
) -> Tensor:
    """Boosts q along the beam axis with given rapidity

    Args:
        q (Tensor): input vector with shape=(b,n,4)/(b,4)
        rapidity (Tensor): boosting parameter with shape=(b,1)/(b,)
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

    q0 = -log(r[:, :, 2] * r[:, :, 3])
    qx = q0 * sqrt(1 - costheta**2) * cos(phi)
    qy = q0 * sqrt(1 - costheta**2) * sin(phi)
    qz = q0 * costheta

    return torch.stack([q0, qx, qy, qz], dim=-1)


def map_fourvector_rambo_diet(q0: Tensor, costheta: Tensor, phi: Tensor) -> Tensor:
    """Transform energies and angles into proper 4-momentum
    Needed for rambo on diet

    Args:
        q0 (Tensor): energy with shape=(b,n-1)
        costheta (Tensor): costheta angle with shape shape=(b,n-1)
        phi (Tensor): azimuthal angle with sshape=(b,n-1)

    Returns:
        q (Tensor): n 4-Momenta with shape=(b,n-1,4)
    """
    qx = q0 * sqrt(1 - costheta**2) * cos(phi)
    qy = q0 * sqrt(1 - costheta**2) * sin(phi)
    qz = q0 * costheta

    return torch.stack([q0, qx, qy, qz], dim=-1)


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


def build_p_in(e_cm: Tensor) -> Tensor:
    """Build symmetric incoming momenta given the center of mass energy

    Args:
        e_cm: Center of mass energy, shape=(b,)

    Returns:
        p_in: Incoming momenta, shape=(b,2,4)
    """
    zeros = torch.zeros_like(e_cm)
    p_cms = e_cm / 2
    p1 = torch.stack([p_cms, zeros, zeros, p_cms], dim=1)
    p2 = torch.stack([p_cms, zeros, zeros, -p_cms], dim=1)
    p_in = torch.stack([p1, p2], dim=1)
    return p_in
