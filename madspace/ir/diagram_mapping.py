from __future__ import annotations
from dataclasses import dataclass, field
from math import pi

from .ir import IRFunction, scalar, four_vector, four_vector_array
from .base import PhaseSpaceMapping, VarList, MapReturn
from .twoparticle import (
    tInvariantTwoParticleCOM,
    tInvariantTwoParticleLAB,
    TwoParticleLAB,
    TwoParticleCOM,
)
from .luminosity import Luminosity, ResonantLuminosity
from .invariants import (
    BreitWignerInvariantBlock,
    UniformInvariantBlock,
    MasslessInvariantBlock,
    StableInvariantBlock,
)


@dataclass(eq=False)
class Line:
    """
    Class describing a line in a Feynman diagram.

    Args:
        mass: mass of the particle, optional, default 0.
        width: decay width of the particle, optional, default 0.
        name: name for the line, optional
    """

    mass: float = 0.0
    width: float = 0.0
    name: str | None = None
    vertices: list[Vertex] = field(init=False, default_factory=list)
    sqrt_s_min: float | None = field(init=False, default=None)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.name is not None:
            return self.name
        elif len(self.vertices) == 2:
            return f"{self.vertices[0]} -- {self.vertices[1]}"
        else:
            return "?"


@dataclass(eq=False)
class Vertex:
    lines: list[Line]
    name: str | None = None

    def __post_init__(self):
        for i, line in enumerate(self.lines):
            line.vertices.append(self)
            if len(line.vertices) > 2:
                raise ValueError(f"Line {i} attached to more than two vertices")

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "?" if self.name is None else self.name


@dataclass
class Diagram:
    incoming: list[Line]
    outgoing: list[Line]
    vertices: list[Vertex]
    t_channel_vertices: list[Vertex] = field(init=False)
    t_channel_lines: list[Line] = field(init=False)
    lines_after_t: list[Line] = field(init=False)
    s_channel_vertices: list[Vertex] = field(init=False)
    s_channel_lines: list[Line] = field(init=False)
    s_decay_layers: list[list[int]] = field(init=False)
    permutation: list[int] = field(init=False)
    inverse_permutation: list[int] = field(init=False)

    def __post_init__(self):
        self._fill_names(self.vertices, "v")
        self._fill_names(self.incoming, "in")
        self._fill_names(self.outgoing, "out")

        (t_channel_lines, self.t_channel_vertices) = self._t_channel_recursive(
            self.incoming[0], None
        )
        self.t_channel_lines = t_channel_lines[1:]
        self._fill_names(self.t_channel_lines, "t")
        self._init_lines_after_t()

        self._init_s_channel()
        self._fill_names(self.s_channel_lines, "s")

    def _fill_names(self, items, prefix):
        for i, item in enumerate(items):
            if item.name is None:
                item.name = f"{prefix}{i+1}"

    def _t_channel_recursive(
        self, line: Line, prev_vertex: Vertex | None
    ) -> tuple[list[Vertex], list[Line]] | None:
        if line is self.incoming[1]:
            return [], []

        if line.vertices[0] is prev_vertex:
            if len(line.vertices) == 1:
                return None
            else:
                vertex = line.vertices[1]
        else:
            vertex = line.vertices[0]

        for out_line in vertex.lines:
            if out_line is line:
                continue
            t_channel = self._t_channel_recursive(out_line, vertex)
            if t_channel is not None:
                return [line, *t_channel[0]], [vertex, *t_channel[1]]
        return None

    def _init_lines_after_t(self):
        t_channel_lines = [self.incoming[0], *self.t_channel_lines, self.incoming[1]]
        self.lines_after_t = []
        for vertex, line_in_1, line_in_2 in zip(
            self.t_channel_vertices, t_channel_lines[:-1], t_channel_lines[1:]
        ):
            for line in vertex.lines:
                if line in [line_in_1, line_in_2]:
                    continue
                self.lines_after_t.append(line)

    def _permutation_recursive(self, lines: list[Line], vertices: list[Vertex]):
        out_indices = []
        for line, parent_vertex in zip(lines, vertices):
            if len(line.vertices) == 1:
                out_indices.append(self.outgoing.index(line))
            else:
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                next_lines = [
                    next_line for next_line in vertex.lines if next_line is not line
                ]
                next_vertices = [vertex] * len(next_lines)
                out_indices.extend(self._permutation_recursive(next_lines, next_vertices))
        return out_indices

    def _init_s_channel(self):
        self.inverse_permutation = self._permutation_recursive(
            self.lines_after_t, self.t_channel_vertices
        )
        self.permutation = [
            self.inverse_permutation.index(i) for i in range(len(self.outgoing))
        ]

        self.s_channel_lines = []
        self.s_channel_vertices = []
        self.s_decay_layers = []
        lines = [self.outgoing[i] for i in self.inverse_permutation]
        vertices = [None] * len(lines)
        has_next_layer = True
        while has_next_layer:
            next_vertices = []
            decay_counts = []
            has_next_layer = False
            for line, parent_vertex in zip(lines, vertices):
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                if vertex not in next_vertices:
                    if vertex in self.t_channel_vertices:
                        next_vertices.append(parent_vertex)
                    else:
                        next_vertices.append(vertex)
                        has_next_layer = True
                    decay_counts.append(0)
                decay_counts[-1] += 1
            self.s_decay_layers.append(decay_counts)

            line_iter = iter(lines)
            vertex_iter = iter(vertices)
            next_lines = []
            for i, (decay_count, vertex) in enumerate(zip(decay_counts, next_vertices)):
                if decay_count == 1:
                    next_vertices[i] = next(vertex_iter)
                    next_lines.append(next(line_iter))
                else:
                    for _ in range(decay_count):
                        next(vertex_iter)
                    decayed_lines = [next(line_iter) for _ in range(decay_count)]
                    next_line = next(
                        line for line in vertex.lines if line not in decayed_lines
                    )
                    next_lines.append(next_line)
                    self.s_channel_lines.append(next_line)
                    self.s_channel_vertices.append(vertex)

            lines = next_lines
            vertices = next_vertices

        del self.s_decay_layers[-1]


class RandomNumbers:
    def __init__(self, random: VarList):
        self.random = random
        self.index = 0

    def __call__(self, count: int = 1) -> IRVariable:
        r = self.random[self.index : self.index + count]
        self.index += count
        return r

    def empty(self) -> bool:
        return self.index == len(self.random)


def ir_sum(ir: IRFunction, terms: list[IRVariable]) -> IRVariable:
    if len(terms) == 0:
        return ir.constant(0.)
    tsum = terms[0]
    for term in terms[1:]:
        tsum = ir.add(tsum, term)
    return tsum


def ir_product(ir: IRFunction, terms: list[IRVariable]) -> IRVariable:
    if len(terms) == 0:
        return ir.constant(1.)
    prod = terms[0]
    for factor in terms[1:]:
        prod = ir.mul(prod, factor)
    return prod


class tDiagramMapping(PhaseSpaceMapping):
    """Implements a mapping for the t-channel part of a Feynman diagram using the algorithm
    described in section 3.2 of
        [1] https://arxiv.org/pdf/2102.00773
    """

    def __init__(self, diagram: Diagram):
        self.n_particles = len(diagram.lines_after_t)
        if self.n_particles != len(diagram.t_channel_vertices):
            raise ValueError(
                "Only vertices with 3 lines are supported in the t-channel part of the diagram"
            )
        self.types_in = [scalar] * (3 * self.n_particles - 4 + 1 + self.n_particles)
        self.types_c = []
        self.types_out = [four_vector] * (self.n_particles + 2)

        none_if_zero = lambda x: None if x == 0 else x

        last_t_line = diagram.t_channel_lines[-1]
        self.t_invariants = [
            tInvariantTwoParticleCOM(flat=True)
            #tInvariantTwoParticleCOM(nu=1.4)
            #if last_t_line.mass == 0.0
            #else tInvariantTwoParticleCOM(
            #    mt=none_if_zero(last_t_line.mass), wt=none_if_zero(last_t_line.width)
            #)
        ]
        self.s_uniform_invariants = []
        for line in reversed(diagram.t_channel_lines[:-1]):
            self.t_invariants.append(
                tInvariantTwoParticleLAB(flat=True)
                #tInvariantTwoParticleLAB(nu=1.4)
                #if line.mass == 0.0
                #else tInvariantTwoParticleLAB(
                #    mt=none_if_zero(line.mass), wt=none_if_zero(line.width)
                #)
            )
            self.s_uniform_invariants.append(UniformInvariantBlock())

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        del condition

        rand = RandomNumbers(inputs[:-self.n_particles-1])
        e_cm = inputs[-self.n_particles-1]
        m_out = inputs[-self.n_particles:]
        dets = []

        # construct initial state momenta
        p1, p2 = ir.com_p_in(e_cm)

        # sample s-invariants from the t-channel part of the diagram
        sqs_max = e_cm
        cumulated_m_out = [m_out[0]]
        for invariant, sqs, sqs_rev in zip(
            self.s_uniform_invariants, m_out[1:-1], m_out[:1:-1]
        ):
            sqs_max = ir.sub(sqs_max, sqs_rev)
            sqs_min = ir.add(cumulated_m_out[-1], sqs)
            s_min = ir.square(sqs_min)
            s_max = ir.square(sqs_max)
            (s,), jac = invariant.map(ir, rand(), condition=[s_min, s_max])
            cumulated_m_out.append(ir.sqrt(s))
            dets.append(jac)

        # sample t-invariants and build momenta of t-channel part of the diagram
        p_out = []
        p2_rest = p2
        for invariant, cum_m_out, mass in zip(
            self.t_invariants, reversed(cumulated_m_out), m_out[:0:-1]
        ):
            (k_rest, k), jac = invariant.map(
                ir, [*rand(2), cum_m_out, mass], condition=[p1, p2_rest]
            )
            p_out.append(k)
            p2_rest = ir.sub_4vec(p2_rest, k)
            dets.append(jac)
        p_out.append(k_rest)
        return (p1, p2, *reversed(p_out)), ir_product(ir, dets)


class DiagramMapping(PhaseSpaceMapping):
    """
    TODO:
        - support quartic vertices
    """

    def __init__(
        self,
        diagram: Diagram,
        s_lab: float,
        s_hat_min: float = 0.0,
        leptonic: bool = False,
        t_mapping: str = "diagram",
        s_min_epsilon: float = 1e-2,
    ):
        n_out = len(diagram.outgoing)
        self.types_in = [scalar] * (3 * n_out - 2 - (2 if leptonic else 0))
        self.types_c = []
        self.types_out = [four_vector_array(n_out + 2)] + [scalar] * 2

        self.diagram = diagram
        self.s_lab = s_lab
        self.sqrt_s_epsilon = s_min_epsilon**0.5
        self.has_t_channel = len(diagram.t_channel_lines) != 0

        # Initialize s invariants and decay mappings
        sqrt_s_min = [
            diagram.outgoing[diagram.inverse_permutation[i]].mass
            for i in range(len(diagram.outgoing))
        ]
        self.s_decay_invariants = []
        self.s_decays = []
        line_iter = iter(diagram.s_channel_lines)
        s_decay_layers = (
            diagram.s_decay_layers if self.has_t_channel else diagram.s_decay_layers[:-1]
        )
        for layer in s_decay_layers:
            sqs_iter = iter(sqrt_s_min)
            sqrt_s_min = []
            layer_invariants = []
            layer_decays = []
            for count in layer:
                sqs_min = max(self.sqrt_s_epsilon, sum(next(sqs_iter) for i in range(count)))
                sqrt_s_min.append(sqs_min)
                if count == 1:
                    continue
                line = next(line_iter)
                layer_invariants.append(
                    MasslessInvariantBlock(nu=1.4)
                    if line.mass == 0.0
                    else (
                        StableInvariantBlock(mass=line.mass, nu=1.4)
                        if line.width == 0.0
                        else BreitWignerInvariantBlock(mass=line.mass, width=line.width)
                    )
                )
                layer_decays.append(TwoParticleLAB())
            self.s_decay_invariants.append(layer_invariants)
            self.s_decays.append(layer_decays)
        if not self.has_t_channel:
            self.s_decay_invariants.append([])
            self.s_decays.append([TwoParticleCOM()])

        # Initialize luminosity and t-channel mapping
        s_hat_min = max(sum(sqrt_s_min) ** 2, s_hat_min)
        self.luminosity = None
        if self.has_t_channel:
            self.t_channel_type = t_mapping
            n_lines_after_t = len(diagram.lines_after_t)
            self.t_random_numbers = 3 * n_lines_after_t - 4
            if not (leptonic or t_mapping == "chili"):
                self.luminosity = Luminosity(s_lab, s_hat_min)
            if t_mapping == "diagram":
                self.t_mapping = tDiagramMapping(diagram)
            elif t_mapping == "rambo":
                self.t_mapping = tRamboBlock(n_lines_after_t)
            elif t_mapping == "chili":
                if leptonic:
                    raise ValueError("chili only supports hadronic processes")
                # TODO: allow to set ymax, ptmin
                self.t_mapping = tChiliBlock(
                    n_lines_after_t,
                    ymax=torch.full((n_lines_after_t,), 4.0),
                    ptmin=torch.full((n_lines_after_t,), 20.0),
                )
                self.t_random_numbers += 2
            else:
                raise ValueError(f"Unknown t-channel mapping {t_mapping}")
        elif not leptonic:
            s_line = diagram.s_channel_lines[-1]
            if s_line.mass != 0.0:
                self.luminosity = ResonantLuminosity(
                    s_lab, s_line.mass, s_line.width, s_hat_min
                )
            else:
                self.luminosity = Luminosity(s_lab, s_hat_min)

        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def _map(self, ir: IRDiagram, inputs: VarList, condition: Varlist) -> MapReturn:
        rand = RandomNumbers(inputs)
        dets = []

        # Do luminosity and get s_hat and rapidity
        if self.luminosity is None:
            x1, x2, s_hat = ir.constant(self.s_lab), ir.constant(1.), ir.constant(1.)
        else:
            (x1, x2, s_hat), jac_lumi = self.luminosity.map(ir, rand(2))
            dets.append(jac_lumi)
            rap = ir.rapidity(x1, x2)
        sqrt_s_hat = ir.sqrt(s_hat)

        # sample s-invariants from decays, starting from the final state particles
        sqrt_s = [ir.constant(line.mass) for line in self.diagram.outgoing]
        decay_masses = []
        decay_s_sqrt_s = []
        for layer_counts, layer_invariants in zip(
            self.diagram.s_decay_layers, self.s_decay_invariants
        ):
            sqrt_s_min = []
            sqrt_s_index = 0
            layer_masses = []
            for decay_count in layer_counts:
                sqs_clip = self.sqrt_s_epsilon if decay_count > 1 else 0.0
                sqrt_s_min.append(
                    ir.clip_min(
                        ir_sum(ir, sqrt_s[sqrt_s_index : sqrt_s_index + decay_count]),
                        sqs_clip,
                    )
                )
                layer_masses.append(sqrt_s[sqrt_s_index : sqrt_s_index + decay_count])
                sqrt_s_index += decay_count
            decay_masses.append(layer_masses)

            if len(layer_invariants) == 0:
                decay_s_sqrt_s.append([(s_hat, sqrt_s_hat)])
                assert not self.has_t_channel
                continue

            sqs_min_sums = [sqrt_s_min[-1]]
            for sqs_min in sqrt_s_min[-2:0:-1]:
                sqs_min_sums.append(ir.add(sqs_min_sums[-1], sqs_min))

            sqs_sum = sqrt_s_hat
            sqrt_s = []
            layer_s_sqrt_s = []
            invariant_iter = iter(layer_invariants)
            for i, decay_count in enumerate(layer_counts):
                if decay_count == 1:
                    sqrt_s.append(sqrt_s_min[i])
                    layer_s_sqrt_s.append((None, None))
                    continue
                s_min = ir.square(sqrt_s_min[i])
                if i == len(layer_counts) - 1:
                    s_max = ir.square(sqs_sum)
                else:
                    s_max = ir.square(ir.sub(sqs_sum, sqs_min_sums[-i - 1]))
                (s,), jac = next(invariant_iter).map(ir, rand(), condition=[s_min, s_max])
                sqs = ir.sqrt(s)
                sqrt_s.append(sqs)
                layer_s_sqrt_s.append((s, sqs))
                sqs_sum = ir.sub(sqs_sum, sqs)
                dets.append(jac)
            decay_s_sqrt_s.append(layer_s_sqrt_s)

        if self.has_t_channel:
            (p1, p2, *p_out), jac = self.t_mapping.map(
                ir, [*rand(self.t_random_numbers), sqrt_s_hat, *sqrt_s]
            )
            if self.t_channel_type == "chili":
                # TODO
                x1 = p_in[:, 0, 0] * 2 / sqrt_s_hat
                x2 = p_in[:, 1, 0] * 2 / sqrt_s_hat
                x1x2 = torch.stack([x1, x2], dim=1)
            dets.append(jac)
        else:
            p1, p2 = ir.com_p_in(sqrt_s_hat)
            p_out = [None]

        # build the momenta of the decays
        for layer_counts, layer_decays, layer_masses, layer_s_sqrt_s in zip(
            reversed(self.diagram.s_decay_layers),
            reversed(self.s_decays),
            reversed(decay_masses),
            reversed(decay_s_sqrt_s),
        ):
            p_out_prev = p_out
            p_out = []
            decay_iter = iter(layer_decays)
            for count, k_in, masses, (dp_s, dp_sqrt_s) in zip(
                layer_counts, p_out_prev, layer_masses, layer_s_sqrt_s
            ):
                if count == 1:
                    p_out.append(k_in)
                    continue
                k_in = [] if k_in is None else [k_in]
                k_out, jac = next(decay_iter).map(
                    ir, [*rand(2), *k_in, dp_s, dp_sqrt_s, *masses]
                )
                p_out.extend(k_out)
                dets.append(jac)

        # we should have consumed all the random numbers
        assert rand.empty()

        # permute and return momenta
        p_out_perm = [p_out[self.diagram.permutation[i]] for i in range(len(p_out))]
        p_ext = ir.stack(p1, p2, *p_out_perm, 0)
        p_ext_lab = p_ext if self.luminosity is None else ir.boost_beam(p_ext, rap)
        ps_weight = ir.mul_const(ir_product(ir, dets), self.pi_factors)
        return (p_ext_lab, x1, x2), ps_weight

