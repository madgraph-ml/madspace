""" Helper functions needed for phase-space mappings """

import torch


MIKOWSKI = torch.diag([1.0, -1.0, -1.0, -1.0])


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
        a (torch.Tensor): 4-vector with shape shape (b,4)
        b (torch.Tensor): 4-vector with shape shape (b,4)
        
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
        k (torch.Tensor): input vector
        p_boost (torch.Tensor): boosting vector

    Returns:
        k' (torch.Tensor): boosted vector
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
