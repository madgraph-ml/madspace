from .ir import IRFunction, scalar
from .base import PhaseSpaceMapping, VarList, MapReturn


class _Invariant(PhaseSpaceMapping):
    types_in = [scalar]
    types_c = [scalar, scalar]
    types_out = [scalar]


class UniformInvariantBlock(_Invariant):
    """Implements uniform sampling of invariants"""

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r = inputs[0]
        s_min, s_max = condition
        s, det = ir.uniform_invariant(r, s_min, s_max)
        return [s], det

    def _map_inverse(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        s = inputs[0]
        s_min, s_max = condition
        r, det = ir.uniform_invariant_inverse(s, s_min, s_max)
        return [r], det


class BreitWignerInvariantBlock(_Invariant):
    """
    Performs the Breit-Wigner mapping as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        mass: float,
        width: float,
    ):
        """
        Args:
            mass: Mass of propagator particle
            width: width of propagator particle
        """
        super().__init__()
        self.mass = mass
        self.width = width

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r = inputs[0]
        s_min, s_max = condition
        mass = ir.constant(mass)
        width = ir.constant(mass)
        s, det = ir.breit_wigner_invariant(r, mass, width, s_min, s_max)
        return [s], det

    def _map_inverse(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        s = inputs[0]
        s_min, s_max = condition
        mass = ir.constant(mass)
        width = ir.constant(mass)
        r, det = ir.breit_wigner_invariant_inverse(s, mass, width, s_min, s_max)
        return [r], det


class StableInvariantBlock(_Invariant):
    """
    Performs the massive, vanishing width propagator as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        mass: float,
        nu: float = 1.4,
    ):
        """
        Args:
            mass: Mass of propagator particle
            nu (optional): controls nu parameter
        """
        self.mass = mass
        self.nu = nu

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r = inputs[0]
        s_min, s_max = condition
        mass = ir.constant(self.mass)
        if self.nu == 1.:
            s, det = ir.stable_invariant(r, mass, s_min, s_max)
        else:
            nu = ir.constant(self.nu)
            s, det = ir.stable_invariant_nu(r, mass, nu, s_min, s_max)
        return [s], det

    def _map_inverse(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r = inputs[0]
        s_min, s_max = condition
        mass = ir.constant(self.mass)
        if self.nu == 1.:
            s, det = ir.stable_invariant_inverse(r, mass, s_min, s_max)
        else:
            nu = ir.constant(self.nu)
            s, det = ir.stable_invariant_nu_inverse(r, mass, nu, s_min, s_max)
        return [s], det


class MasslessInvariantBlock(StableInvariantBlock):
    """
    Performs the massless propagator as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        nu: float = 1.4,
    ):
        """
        Args:
            nu (float, optional): controls nu parameter
        """
        super().__init__(mass=0., nu=nu)
