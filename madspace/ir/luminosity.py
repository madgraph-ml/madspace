from .ir import IRFunction, IRVariable, scalar
from .base import PhaseSpaceMapping, VarList, MapReturn


class _Luminosity(PhaseSpaceMapping):
    """
    Implements a base luminosity mapping
    """
    types_in = [scalar, scalar]
    types_c = []
    types_out = [scalar, scalar, scalar]

    def __init__(
        self, s_lab: float, shat_min: float, shat_max: float = None,
    ):
        """
        Args:
            s_lab: squared COM energy of the lab frame
            shat_min: minimum s_hat
            shat_max: maximum s_hat. Defaults to None. None means s_max = s_lab.
        """
        self.s_lab = s_lab
        self.shat_min = shat_min
        self.shat_max = shat_max if shat_max is not None else s_lab

    def _shat_map(self, ir: IRFunction, r1: IRVariable):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide shat_map(...) method"
        )

    def _shat_map_inverse(self, ir: IRFunction, shat: IRVariable):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide shat_map_inverse(...) method"
        )

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        del condition
        r1, r2 = inputs
        shat, s_det = self._shat_map(ir, r1)
        x1, x2, x_det = ir.r_to_x1x2(r2, shat, self.s_lab)
        det = ir.mul(s_det, x_det)
        return [x1, x2, shat], det

    def _map_inverse(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        del condition
        x1, x2, shat = inputs
        r2, x_det = ir.x1x2_to_r(x1, x2, self.s_lab)
        r1, s_det = self._shat_map_inverse(ir, shat)
        det = ir.mul(s_det, x_det)
        return [r1, r2], det


class Luminosity(_Luminosity):
    """
    Implement luminosity mapping as suggested in
        [1] https://freidok.uni-freiburg.de/data/154629
        [2] https://arxiv.org/abs/hep-ph/0206070v2
        [3] https://arxiv.org/abs/hep-ph/0008033

    which aims to smooth the overall 1/s dependency of the
    partonic cross section (due to the flux factor)

    ### Note:
    In contrast to [1] we also allow for an tau_max (or s_max)
    to potentially split up the luminosity integral into different regions
    as been done by MG5. Then, we use the massless mapping as suggested
    in [1-3]. This slightly alters the map as:

        [1:lumi] Eq.(H.5)
            tau = tau_min ** r1
        [1-3:massless]
            tau = tau_max ** r1 * tau_min ** (1-r1)

    result in some slight difference in the determinant.

    """

    def __init__(
        self, s_lab: float, shat_min: float, shat_max: float = None, nu: float = 1.0
    ):
        """
        Args:
            nu (float): nu parameter from [1-3]
        """
        super().__init__(s_lab, shat_min, shat_max)
        self.nu = nu

    def _shat_map(self, ir: IRFunction, r1: IRVariable):
        if self.nu == 1:
            return ir.stable_invariant(
                r1, ir.constant(0.), ir.constant(self.shat_min), ir.constant(self.shat_max)
            )
        else:
            return ir.stable_invariant_nu(
                r1,
                ir.constant(0.),
                ir.constant(self.nu),
                ir.constant(self.shat_min),
                ir.constant(self.shat_max),
            )

    def _shat_map_inverse(self, ir: IRFunction, shat: IRVariable):
        if self.nu == 1:
            return ir.stable_invariant_inverse(
                shat, ir.constant(0.), ir.constant(self.shat_min), ir.constant(self.shat_max)
            )
        else:
            return ir.stable_invariant_nu_inverse(
                shat,
                ir.constant(0.),
                ir.constant(self.nu),
                ir.constant(self.shat_min),
                ir.constant(self.shat_max),
            )


class ResonantLuminosity(_Luminosity):
    """
    Implement luminosity mapping that takes into account resonant
    particles in the s-channel related to the partonic COM
    energy s_hat.

    As this might only be used in a regime s_hat in [M-n\Gamma, M+n\Gamma],
    this translates in both lower (tau_min) and upper limits (tau_max).
    """

    def __init__(
        self,
        s_lab: float,
        mass: float,
        width: float,
        shat_min: float,
        shat_max: float = None,
    ):
        """
        Args:
            mass
            width
        """
        super().__init__(s_lab, shat_min, shat_max)
        self.mass = mass
        self.width = width

    def _shat_map(self, ir: IRFunction, r1: IRVariable):
        return ir.breit_wigner_invariant(
            r1,
            ir.constant(self.mass),
            ir.constant(self.width),
            ir.constant(self.shat_min),
            ir.constant(self.shat_max),
        )

    def _shat_map_inverse(self, ir: IRFunction, shat: IRVariable):
        return ir.breit_wigner_invariant_inverse(
            shat,
            ir.constant(self.mass),
            ir.constant(self.width),
            ir.constant(self.shat_min),
            ir.constant(self.shat_max),
        )


class FlatLuminosity(_Luminosity):
    """
    Implement luminosity mapping which maps out tau flat:
        tau = tau_min + (tau_max - tau_min) * r1
    """

    def _shat_map(self, ir: IRFunction, r1: IRVariable):
        s = ir.uniform(r1, self.shat_min, self.shat_max)
        det = ir.constant(self.shat_max - self.shat_min)
        return s, det

    def _shat_map_inverse(self, ir: IRFunction, shat: IRVariable):
        r = ir.uniform_inverse(shat, self.shat_min, self.shat_max)
        det = ir.constant(1 / (self.shat_max - self.shat_min))
        return r, det


