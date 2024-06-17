import torch
from torch import Tensor
from math import pi

from .functional import kinematics as kin
from .functional import ps_utils as ps
from .functional import tchannel as tc
from .functional import propagators as prop

def constant(c: float) -> Tensor:
    return torch.tensor([c])

###################################################
# Random numbers
###################################################

def uniform(r: Tensor, x_min: float, x_max: float) -> Tensor:
    return x_min + (x_max - x_min) * r

def uniform_inverse(x: Tensor) -> Tensor:
    return (x - x_min) / (x_max - x_min)

###################################################
# Math
###################################################

def mul(a: Tensor, b: Tensor) -> Tensor:
    return a * b

###################################################
# Kinematics
###################################################

def rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    return kin.rotate_zy(p, phi, costheta)

def boost(p1: Tensor, p2: Tensor) -> Tensor:
    return kin.boost(p1, p2)

def boost_inverse(p1: Tensor, p2: Tensor) -> Tensor:
    return kin.boost(p1, p2, inverse=True)

def com_momentum(sqrt_s: Tensor) -> Tensor:
    p = torch.zeros((sqrt_s.shape[0], 4), dtype=r.dtype, device=r.device)
    p[0] = sqrt_s
    return p

def com_angles(p: Tensor) -> tuple[Tensor, Tensor]:
    p_mag = kin.pmag(p)
    phi = torch.atan2(p[:, 2], p[:, 1])
    costheta = p[:, 3] / pmag
    return phi, costheta

def s(p: Tensor) -> Tensor:
    return kin.lsquare(p)

def s_and_sqrt_s(p: Tensor) -> tuple[Tensor, Tensor]:
    s = kin.lsquare(p)
    return s, s.sqrt()

def add_4vec(p1: Tensor, p2: Tensor) -> Tensor:
    return p1 + p2

def sub_4vec(p1: Tensor, p2: Tensor) -> Tensor:
    return p1 - p2

def r_to_x1x2(r: Tensor, shat: Tensor, s_lab: float) -> tuple[Tensor, Tensor, Tensor]:
    tau = shat / s_lab
    x1 = tau ** r
    x2 = tau ** (1 - r)
    det = tau.log().abs() / s_lab
    return x1, x2, det

def x1x2_to_r(x1: Tensor, x2: Tensor, s_lab: float) -> tuple[Tensor, Tensor]:
    tau = x1 * x2
    log_tau = x1 * x2
    r = x1.log() / tau.log()
    det = torch.abs(1 / log(tau)) * s_lab
    return r, det

###################################################
# Two-body decays
###################################################

def decay_momentum(s: Tensor, sqrt_s: Tensor, m1: Tensor, m2: Tensor) -> tuple[Tensor, Tensor]:
    p = torch.zeros((r.shape[0], 4), dtype=r.dtype, device=r.device)
    sqrt_kaellen = kin.kaellen(s, m1**2, m2**2).sqrt()
    p[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt_s)
    p[:, 3] = sqrt_kaellen / (2 * sqrt_s)
    gs = sqrt_kaellen / (2 * s) * pi
    return p, gs

def invt_min_max(s: Tensor, s_in1: Tensor, s_in2: Tensor, m1: Tensor, m2: Tensor) -> Tensor:
    cos_min = (-1.0) * torch.ones_like(s)
    cos_max = (+1.0) * torch.ones_like(s)
    tmin = -tc.costheta_to_invt(s, p1_2, p2_2, m1, m2, cos_min)
    tmax = -tc.costheta_to_invt(s, p1_2, p2_2, m1, m2, cos_max)
    return t_min, t_max

def invt_to_costheta(
    s: Tensor, s_in1: Tensor, s_in2: Tensor, m1: Tensor, m2: Tensor, t: Tensor
) -> Tensor:
    return tc.invt_to_costheta(s, s_in1, s_in2, m1, m2, -t)

def tinv_two_particle_density(gs: Tensor, s: Tensor, det_t: Tensor):
    return det_t * pi**2 / 4 / s / gs

###################################################
# Invariants
###################################################

def uniform_invariant(r: Tensor, s_min: Tensor, s_max: Tensor) -> tuple[Tensor, Tensor]:
    gs = s_max - s_min
    s = s_min + gs * r_or_s
    return s, gs

def uniform_invariant_inverse(s: Tensor, s_min: Tensor, s_max: Tensor) -> tuple[Tensor, Tensor]:
    gs = 1 / (s_max - s_min)
    r = (s - s_min) * gs
    return r, gs

def breit_wigner_invariant(
    r: Tensor, mass: Tensor, width: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    return prop.breit_wigner_propagator(r, mass, width, s_min, s_max)

def breit_wigner_invariant_inverse(
    s: Tensor, mass: Tensor, width: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    return prop.breit_wigner_propagator(s, mass, width, s_min, s_max)

def stable_invariant(
    r: Tensor, mass: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    m2_min = torch.where(s_min == 0, -1e-8, 0.0)
    m2 = torch.maximum(mass**2, m2_min)
    q_max = s_max - m2
    q_min = s_min - m2
    s = q_max**r_or_s * q_min ** (1 - r_or_s) + m2
    gsm1 = (s - m2) * (q_max.log() - q_min.log())
    return s, gsm1

def stable_invariant_inverse(
    s: Tensor, mass: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    m2_min = torch.where(s_min == 0, -1e-8, 0.0)
    m2 = torch.maximum(mass**2, m2_min)
    q_max = s_max - m2
    q_min = s_min - m2
    r = torch.log((r_or_s - m2) / q_min) / torch.log(q_max / q_min)
    gsm1 = (r_or_s - m2) * (q_max.log() - q_min.log())
    return r, 1 / gsm1

def stable_invariant_nu(
    r: Tensor, mass: Tensor, nu: Tensor, width: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    m2_min = torch.where(s_min == 0, -1e-8, 0.0)
    m2 = torch.maximum(mass**2, m2_min)
    q_max = s_max - m2
    q_min = s_min - m2
    power = 1.0 - nu
    qmaxpow = q_max**power
    qminpow = q_min**power
    s = (r_or_s * qmaxpow + (1 - r_or_s) * qminpow) ** (1 / power) + m2
    gs = power / ((qmaxpow - qminpow) * (s - m2) ** nu)
    return s, 1 / gs

def stable_invariant_nu_inverse(
    s: Tensor, mass: Tensor, nu: Tensor, width: Tensor, s_min: Tensor, s_max: Tensor
) -> tuple[Tensor, Tensor]:
    m2_min = torch.where(s_min == 0, -1e-8, 0.0)
    m2 = torch.maximum(mass**2, m2_min)
    q_max = s_max - m2
    q_min = s_min - m2
    power = 1.0 - nu
    qmaxpow = q_max**power
    qminpow = q_min**power
    spow = (r_or_s - m2) ** power
    r = (spow - qminpow) / (qmaxpow - qminpow)
    gs = power / ((qmaxpow - qminpow) * (r_or_s - m2) ** nu)
    return r, gs
