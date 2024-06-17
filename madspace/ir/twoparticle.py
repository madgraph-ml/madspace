from math import pi

from .ir import IRFunction, scalar, four_vector
from .base import PhaseSpaceMapping, VarList, MapReturn
from .invariants import (
    BreitWignerInvariantBlock,
    UniformInvariantBlock,
    MasslessInvariantBlock,
    StableInvariantBlock,
)


class TwoParticleCOM(PhaseSpaceMapping):
    types_in = [scalar, scalar, scalar, scalar, scalar, scalar]
    types_c = []
    types_out = [four_vector, four_vector]

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r1, r2, s, sqrt_s, m1, m2 = inputs
        phi = ir.uniform(r1, -pi, pi)
        costheta = ir.uniform(r2, -1, 1)
        p0 = ir.com_momentum(sqrt_s)
        p1, gs = ir.decay_momentum(s, sqrt_s, m1, m2)
        p1 = ir.rotate_zy(p1, phi, costheta)
        p2 = ir.sub_4vec(p0, p1)
        return [p1, p2], gs


class TwoParticleLAB(PhaseSpaceMapping):
    types_in = [scalar, scalar, four_vector, scalar, scalar, scalar, scalar]
    types_c = []
    types_out = [four_vector, four_vector]

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        del condition
        r1, r2, p0, s, sqrt_s, m1, m2 = inputs
        phi = ir.uniform(r1, -pi, pi)
        costheta = ir.uniform(r2, -1, 1)
        p1, gs = ir.decay_momentum(s, sqrt_s, m1, m2)
        p1 = ir.rotate_zy(p1, phi, costheta)
        p1 = ir.boost(p1, p0)
        p2 = ir.sub_4vec(p0, p1)
        return [p1, p2], gs


class tInvariantTwoParticleCOM(PhaseSpaceMapping):
    types_in = [scalar, scalar, scalar, scalar]
    types_c = [four_vector, four_vector]
    types_out = [four_vector, four_vector]

    def __init__(
        self,
        mt: float | None = None,
        wt: float | None = None,
        nu: float = 1.4,
        flat: bool = False,
    ):
        if flat:
            self.t_map = UniformInvariantBlock()
        elif mt is None:
            self.t_map = MasslessInvariantBlock(nu=nu)
        elif wt is None:
            self.t_map = StableInvariantBlock(mass=mt, nu=nu)
        else:
            self.t_map = BreitWignerInvariantBlock(mass=mt, width=wt)

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r1, r2, m1, m2 = inputs
        p_in1, p_in2 = condition
        p_tot = ir.add_4vec(p_in1, p_in2)
        s, sqrt_s = ir.s_and_sqrt_s(p_tot)
        s_in1 = ir.s(p_in1)
        s_in2 = ir.s(p_in2)
        t_min, t_max = ir.invt_min_max(s, s_in1, s_in2, m1, m2)
        (t,), det_t = self.t_map.map(ir, [r2], condition=[t_max, t_min])
        phi = ir.uniform(r1, -pi, pi)
        costheta = ir.invt_to_costheta(s, s_in1, s_in2, m1, m2)
        p1, gs = ir.decay_momentum(s, sqrt_s, m1, m2)
        phi1, costheta1 = ir.com_angles(p_in1)
        p1 = ir.rotate_zy(p1, phi, costheta)
        p1 = ir.rotate_zy(p1, phi1, costheta1)
        p2 = ir.sub_4vec(p_tot, p1)
        det = ir.tinv_two_particle_density(gs, s, det_t)
        return [p1, p2], det


class tInvariantTwoParticleLAB(PhaseSpaceMapping):
    types_in = [scalar, scalar, scalar, scalar]
    types_c = [four_vector, four_vector]
    types_out = [four_vector, four_vector]

    def __init__(
        self,
        mt: float | None = None,
        wt: float | None = None,
        nu: float = 1.4,
        flat: bool = False,
    ):
        if flat:
            self.t_map = UniformInvariantBlock()
        elif mt is None:
            self.t_map = MasslessInvariantBlock(nu=nu)
        elif wt is None:
            self.t_map = StableInvariantBlock(mass=mt, nu=nu)
        else:
            self.t_map = BreitWignerInvariantBlock(mass=mt, width=wt)

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        r1, r2, m1, m2 = inputs
        p_in1, p_in2 = condition
        p_tot = ir.add_4vec(p_in1, p_in2)
        s, sqrt_s = ir.s_and_sqrt_s(p_tot)
        s_in1 = ir.s(p_in1)
        s_in2 = ir.s(p_in2)
        p_in1_com = ir.boost_inverse(p_in1, p_tot)
        t_min, t_max = ir.invt_min_max(s, s_in1, s_in2, m1, m2)
        (t,), det_t = self.t_map.map(ir, [r2], condition=[t_max, t_min])
        phi = ir.uniform(r1, -pi, pi)
        costheta = ir.invt_to_costheta(s, s_in1, s_in2, m1, m2)
        p1, gs = ir.decay_momentum(s, sqrt_s, m1, m2)
        phi1, costheta1 = ir.com_angles(p_in1_com)
        p1 = ir.rotate_zy(p1, phi, costheta)
        p1 = ir.rotate_zy(p1, phi1, costheta1)
        p1 = ir.boost(p1, p_tot)
        p2 = ir.sub_4vec(p_tot, p1)
        det = ir.tinv_two_particle_density(gs, s, det_t)
        return [p1, p2], det
