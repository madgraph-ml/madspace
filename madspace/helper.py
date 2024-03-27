""" Helper functions needed for phase-space mappings """

from typing import Callable, Optional, Union
import torch
from math import pi


MIKOWSKI = torch.diag([1.0, -1.0, -1.0, -1.0])
ITER = 100
XTOL = 2e-12
RTOL = 4 * torch.finfo(float).eps


def kaellen(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Definition of the standard kaellen function [1]
    
    [1] https://en.wikipedia.org/wiki/Källén_function

    Args:
        a (torch.Tensor): input 1
        b (torch.Tensor): input 2
        c (torch.Tensor): input 3

    Returns:
        torch.Tensor: Kaellen function
    """
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a


def rotate_zy(p: torch.Tensor, phi: torch.Tensor, theta: torch.Tensor):
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
        p (torch.Tensor): 4-momentum to rotate
        phi (torch.Tensor): rotation angle phi
        theta (torch.tensor): rotation angle theta

    Returns:
        p' (torch.Tensor): Rotated vector
    """
    
    # Define the rotation
    q0 = p[:, 0]
    q1 = p[:, 1] * torch.cos(theta) * torch.cos(phi) \
        + p[:, 3] * torch.sin(theta) * torch.cos(phi) \
        - p[:, 2] * torch.sin(phi)
    q2 = p[:, 1] * torch.cos(theta) * torch.sin(phi) \
        + p[:, 3] * torch.sin(theta) * torch.sin(phi) \
        + p[:, 2] * torch.cos(phi)
    q3 = p[:, 3] * torch.cos(theta) - p[:, 1] * torch.sin(theta)

    return torch.stack((q0, q1, q2, q3), dim=-1)

def lsquare(a: torch.Tensor):
    """Gives the lorentz invariant a^2 using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)
    
    Args:
        a (torch.Tensor): 4-vector with shape shape (b,4)
        
    Returns:
        torch.Tensor: Lorentzscalar with shape (b,)
    """
    return torch.einsum("bd,dd,bd->b", a, MIKOWSKI, a)

def ldot(a: torch.Tensor, b: torch.Tensor):
    """Gives the Lorentz inner product ab using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)
    
    Args:
        a (torch.Tensor): 4-vector with shape shape (b,4)
        b (torch.Tensor): 4-vector with shape shape (b,4)
        
    Returns:
        torch.Tensor: Lorentzscalar with shape (b,)
    """
    return torch.einsum("bd,dd,bd->b", a, MIKOWSKI, b)

def edot(a: torch.Tensor, b: torch.Tensor):
    """Gives the euclidean inner product ab using
    the Euclidean metric
    
    Args:
        a (torch.Tensor): 4-vector with shape (b,4)
        b (torch.Tensor): 4-vector with shape (b,4)
        
    Returns:
        torch.Tensor: Lorentzscalar with shape (b,)
    """
    return torch.einsum("bd,bd->b", a, b)

def boost(k: torch.Tensor, p_boost: torch.Tensor):
    """
    Boost k into the frame of p_boost in argument.
    This means that the following command, for any vector k=(E, px, py, pz)
    gives:
    
        k  -> k' = boost(k, -k) = (M,0,0,0)
        k' -> k  = boost(k', k) = (E, px, py, pz)

    Args:
        k (torch.Tensor): input vector with shape (b,4)
        p_boost (torch.Tensor): boosting vector with shape (b,4)

    Returns:
        k' (torch.Tensor): boosted vector with shape (b,4)
    """
    # Make sure energy is > 0 even after momentum flip (for inverse boost)
    p_boost[:, 0] = torch.abs(p_boost[:, 0])
    
    # Perform the boost
    rsq = torch.sqrt(lsquare(p_boost))         
    k0 = edot(k, p_boost) / rsq
    c1 = (k[:, 0] + k0) / (rsq + p_boost[:, 0])
    k1 = k[:,1] + c1 * p_boost[:,1]
    k2 = k[:,2] + c1 * p_boost[:,2]
    k3 = k[:,3] + c1 * p_boost[:,3]
    
    return torch.stack((k0, k1, k2, k3), dim=-1)

def boost_beam(q, rapidity, inverse=False):
    sign = -1.0 if inverse else 1.0

    pi0 = q[:, :, 0] * torch.cosh(rapidity) + sign * q[:, :, 3] * torch.sinh(rapidity)
    pix = q[:, :, 1]
    piy = q[:, :, 2]
    piz = q[:, :, 3] * torch.cosh(rapidity) + sign * q[:, :, 0] * torch.sinh(rapidity)
    p = torch.stack((pi0, pix, piy, piz), axis=-1)

    return p

def map_fourvector_rambo(xs: torch.Tensor) -> torch.Tensor:
    """Transform unit hypercube points into into four-vectors."""
    cos = 2.0 * xs[:, :, 0] - 1.0
    phi = 2.0 * pi * xs[:, :, 1]

    q = torch.zeros_like(xs)
    q[:, :, 0] = -torch.log(xs[:, :, 2] * xs[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * torch.sqrt(1 - cos**2) * torch.cos(phi)
    q[:, :, 2] = q[:, :, 0] * torch.sqrt(1 - cos**2) * torch.sin(phi)
    q[:, :, 3] = q[:, :, 0] * cos

    return q


def two_body_decay_factor(
    M_i_minus_1: torch.Tensor,
    M_i: torch.Tensor,
    m_i_minus_1: torch.Tensor,
) -> torch.Tensor:
    """Gives two-body decay factor from recursive n-body phase space"""
    return (
        1.0
        / (8 * M_i_minus_1**2)
        * torch.sqrt(
            (M_i_minus_1**2 - (M_i + m_i_minus_1) ** 2)
            * (M_i_minus_1**2 - (M_i - m_i_minus_1) ** 2)
        )
    )
    
def rambo_func(
    x: Union[float, torch.Tensor],
    nparticles: int,
    xs: torch.Tensor,
    diff: bool = False,
) -> torch.Tensor:
    if isinstance(x, float):
        x = x * torch.ones_like(xs[:, 0 : nparticles - 2])
    elif isinstance(x, torch.Tensor):
        assert x.shape[1] == nparticles - 2
    else:
        raise ValueError("x is not valid input")

    i = torch.arange(2, nparticles)[None, :]
    f = (
        (nparticles + 1 - i) * x ** (2 * (nparticles - i))
        - (nparticles - i) * x ** (2 * (nparticles + 1 - i))
        - xs[:, 0 : nparticles - 2]
    )
    if diff:
        df = (nparticles + 1 - i) * (2 * (nparticles - i)) * x ** (
            2 * (nparticles - i) - 1
        ) - (nparticles - i) * (2 * (nparticles + 1 - i)) * x ** (
            2 * (nparticles + 1 - i) - 1
        )
        return df
    return f


def mass_func(
    x: Union[float, torch.Tensor],
    p: torch.Tensor,
    m: torch.Tensor,
    e_cm: torch.Tensor,
    diff: bool = False,
) -> torch.Tensor:
    if isinstance(x, float):
        x = x * torch.ones(m.shape[0], 1)
    elif isinstance(x, torch.Tensor):
        assert x.dim() == 1
    else:
        raise ValueError("x is not valid input")

    root = torch.sqrt(x[:, None] ** 2 * p[:, :, 0] ** 2 + m**2)
    f = torch.sum(root, dim=-1) - e_cm[:, 0]
    if diff:
        return torch.sum(x[:, None] * p[:, :, 0] ** 2 / root, dim=-1)
    return f


def newton(
    f: Callable,
    df: Callable,
    a: float,
    b: float,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = ITER,
    epsilon=1e-8,
):
    if torch.any(f(a) * f(b) > 0):
        raise ValueError(f"None or no unique root in given intervall [{a},{b}]")
    
    # Define lower/upper boundaries as tensor
    xa = a * torch.ones_like(f(x0))
    xb = b * torch.ones_like(f(x0))

    # initilize guess
    if x0 is None:
        x0 = (xa + xb) / 2

    for _ in range(max_iter):
        if torch.any(df(x0) < epsilon):
            raise ValueError("Derivative is too small")

        # do newtons-step
        x1 = x0 - f(x0) / df(x0)

        # check if within given intervall
        higher = x1 > xb
        lower = x1 < xa
        if torch.any(higher):
            x1[higher] = (xb[higher] + x0[higher]) / 2
        if torch.any(lower):
            x1[lower] = (xa[lower] + x0[lower]) / 2

        if torch.allclose(x1, x0, atol=XTOL, rtol=RTOL):
            return x1
        
        # Adjust brackets
        low = f(x1) * f(xa) > 0
        xa[low] = x1[low]
        xb[~low] = x1[~low]

        x0 = x1

    print(f"not converged")
    return x0
