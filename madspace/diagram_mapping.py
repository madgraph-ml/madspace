from __future__ import annotations
from dataclasses import dataclass, field
from math import pi

import torch

from .base import PhaseSpaceMapping, TensorList
from .helper import boost_beam, lsquare, build_p_in
from .twoparticle import (
    tInvariantTwoParticleCOM,
    tInvariantTwoParticleLAB,
    TwoParticleLAB,
    TwoParticleCOM,
)
from .luminosity import Luminosity, ResonantLuminosity
from .invariants import (
    BreitWignerInvariantBlock, UniformInvariantBlock, MasslessInvariantBlock, StableInvariantBlock
)

from icecream import ic

@dataclass(eq=False)
class Line:
    """
    Class describing a line in a Feynman diagram.

    Args:
        mass: mass of the particle, optional, default 0.
        width: decay width of the particle, optional, default 0.
        name: name for the line, optional
    """
    mass: float = 0.
    width: float = 0.
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

        (
            t_channel_lines, self.t_channel_vertices
        ) = self._t_channel_recursive(self.incoming[0], None)
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

    def _init_s_channel(self):
        lines = self.lines_after_t
        vertices = self.t_channel_vertices

        self.s_channel_lines = []
        self.s_channel_vertices = []
        self.s_decay_layers = []
        has_next_layer = True
        while has_next_layer:
            next_lines = []
            next_vertices = []
            layer = []
            has_next_layer = False
            for line, parent_vertex in zip(lines, vertices):
                if line in self.outgoing:
                    next_lines.append(line)
                    next_vertices.append(parent_vertex)
                    layer.append(1)
                    continue
                self.s_channel_lines.append(line)
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                self.s_channel_vertices.append(vertex)

                decay_count = 0
                for next_line in vertex.lines:
                    if next_line is line:
                        continue
                    next_lines.append(next_line)
                    has_next_layer = True
                    next_vertices.append(vertex)
                    decay_count += 1
                layer.append(decay_count)
            lines = next_lines
            vertices = next_vertices
            self.s_decay_layers.append(layer)
        del self.s_decay_layers[-1]

        self.permutation = [lines.index(line) for line in self.outgoing]
        self.inverse_permutation = [self.outgoing.index(line) for line in lines]


class RandomNumbers:
    def __init__(self, random: Tensor):
        self.random = random
        self.index = 0

    def __call__(self, count: int = 1) -> Tensor:
        r = self.random[:, self.index : self.index + count]
        self.index += count
        return r

    def empty(self) -> bool:
        return self.index == self.random.shape[1]


class tDiagramMapping(PhaseSpaceMapping):
    """Implements a mapping for the t-channel part of a Feynman diagram using the algorithm
    described in section 3.2 of
        [1] https://arxiv.org/pdf/2102.00773
    """
    def __init__(self, diagram: Diagram):
        n_particles = len(diagram.lines_after_t)
        if n_particles != len(diagram.t_channel_vertices):
            raise ValueError(
                "Only vertices with 3 lines are supported in the t-channel part of the diagram"
            )
        self.n_random = 3 * n_particles - 4
        dims_in = [(self.n_random,), (), (n_particles,)]
        dims_out = [(n_particles, 4)]
        super().__init__(dims_in, dims_out)

        none_if_zero = lambda x: None if x == 0 else x

        last_t_line = diagram.t_channel_lines[-1]
        self.t_invariants = [
            tInvariantTwoParticleCOM(nu=1.4)
            if last_t_line.mass == 0. else
            tInvariantTwoParticleCOM(
                mt=none_if_zero(last_t_line.mass), wt=none_if_zero(last_t_line.width)
            )
        ]
        self.s_uniform_invariants = []
        for line in reversed(diagram.t_channel_lines[:-1]):
            self.t_invariants.append(
                tInvariantTwoParticleLAB(nu=1.4)
                if line.mass == 0. else
                tInvariantTwoParticleLAB(
                    mt=none_if_zero(line.mass), wt=none_if_zero(line.width)
                )
            )
            self.s_uniform_invariants.append(UniformInvariantBlock())

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm, m_out]
                r: random numbers with shape=(b,3*n-4)
                e_cm: COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles with shape=(b,n)

        Returns:
            p_in (Tensor): incoming momenta with shape=(b,2,4)
            p_out (Tensor): output momenta with shape=(b,n,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition

        rand = RandomNumbers(inputs[0])  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()
        m_out = inputs[2]  # has dims (b,n)
        det = 1.

        # construct initial state momenta
        p_in = build_p_in(e_cm)
        p1, p2 = p_in[:,0], p_in[:,1]

        # sample s-invariants from the t-channel part of the diagram
        sqrt_s_max = e_cm[:,None] - m_out.flip([1])[:, :-2].cumsum(dim=1)
        cumulated_m_out = [m_out[:,:1]]
        for invariant, sqs, sqs_max in zip(
            self.s_uniform_invariants, m_out[:, 1:-1].unbind(dim=1), sqrt_s_max.unbind(dim=1)
        ):
            s_min = (cumulated_m_out[-1] + sqs[:,None]) ** 2
            s_max = sqs_max[:,None] ** 2
            (s, ), jac = invariant.map([rand()], condition=[s_min, s_max])
            cumulated_m_out.append(s.sqrt())
            det *= jac

        # sample t-invariants and build momenta of t-channel part of the diagram
        p_out = []
        p_t_in = p_in
        p2_rest = p2
        for invariant, cum_m_out, mass in zip(
            self.t_invariants, reversed(cumulated_m_out), reversed(m_out[:, 1:].unbind(dim=1))
        ):
            m_t = torch.cat([cum_m_out, mass[:, None]], dim=1)
            (ks, ), jac = invariant.map([rand(2), m_t], condition=[p_t_in])
            k_rest, k = ks[:, 0], ks[:, 1]
            p_out.append(k)
            p2_rest = p2_rest - k
            p_t_in = torch.stack([p1, p2_rest], dim=1)
            det *= jac
        p_out.append(k_rest)
        p_out = torch.stack(p_out, dim=1).flip([1])
        return (p_in, p_out,), det


class DiagramMapping(PhaseSpaceMapping):
    """
    TODO:
        - support quartic vertices
        - alternative strategy: chili + s-channel
        - alternative strategy: rambo + s-channel
    """
    def __init__(
        self,
        diagram: Diagram,
        s_lab: Tensor,
        s_hat_min: float = 0.,
        leptonic: bool = False,
        t_mapping: str = "diagram",
        s_min_epsilon: float = 1e-2,
    ):
        n_out = len(diagram.outgoing)
        dims_in = [(3 * n_out - 2 - (0 if leptonic else 2), )]
        dims_out = [(n_out, 4), (2, )]
        super().__init__(dims_in, dims_out)

        self.diagram = diagram
        self.s_lab = s_lab
        self.sqrt_s_epsilon = s_min_epsilon ** 0.5

        epsilons = [0.] * len(diagram.outgoing)
        for layer in reversed(diagram.s_decay_layers):
            eps_iter = iter(epsilons)
            epsilons = []
            for count in layer:
                epsilons.append(
                    max(self.sqrt_s_epsilon, sum(next(eps_iter) for i in range(count)))
                )
        s_min_decay = sum(epsilons) ** 2

        s_hat_min = torch.tensor(max(
            sum(line.mass for line in diagram.outgoing) ** 2, s_min_decay, s_hat_min
        ))

        # Initialize luminosity and t-channel mapping
        self.has_t_channel = len(diagram.t_channel_lines) != 0
        self.luminosity = None
        if self.has_t_channel:
            if not (leptonic or t_mapping == "chili"):
                self.luminosity = Luminosity(s_lab, s_hat_min)
            if t_mapping == "diagram":
                self.t_mapping = tDiagramMapping(diagram)
            else:
                raise ValueError(f"Unknown t-channel mapping {t_mapping}")
        elif not leptonic:
            s_line = diagram.s_channel_lines[0]
            if self.diagram.s_channel_lines[0] is not None:
                self.luminosity = ResonantLuminosity(
                    s_lab, s_line.mass, s_line.width, s_hat_min
                )
            else:
                self.luminosity = Luminosity(s_lab, s_hat_min)

        # Initialize s invariants and decay mappings
        self.s_decay_invariants = []
        self.s_decays = []
        line_iter = iter(diagram.s_channel_lines)
        is_com_decay = not self.has_t_channel
        for layer in diagram.s_decay_layers:
            layer_invariants = []
            layer_decays = []
            for count in layer:
                if count == 1:
                    continue
                line = next(line_iter)
                if is_com_decay:
                    layer_decays.append(TwoParticleCOM())
                    is_com_decay = False
                    continue
                layer_invariants.append(
                    MasslessInvariantBlock(nu=1.4)
                    if line.mass == 0. else
                    (
                        StableInvariantBlock(mass=line.mass, nu=1.4)
                        if line.width == 0. else
                        BreitWignerInvariantBlock(mass=line.mass, width=line.width)
                    )
                )
                layer_decays.append(TwoParticleLAB())
            self.s_decay_invariants.append(layer_invariants)
            self.s_decays.append(layer_decays)

        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        random = inputs[0]
        rand = RandomNumbers(random)
        ps_weight = 1.

        # Do luminosity and get s_hat and rapidity
        if self.luminosity is None:
            s_hat = torch.full((random.shape[0],), self.s_lab, device=random.device)
            det_lumi = 1.0
            x1x2 = torch.ones((random.shape[0], 2), device=random.device)
        else:
            (x1x2,), jac_lumi = self.luminosity.map([rand(2)])
            ps_weight *= jac_lumi
            s_hat = self.s_lab * x1x2.prod(dim=1)
            rap = 0.5 * torch.log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        sqrt_s_hat = s_hat.sqrt()

        # sample s-invariants from decays, starting from the final state particles
        sqrt_s = [torch.full_like(sqrt_s_hat, line.mass)[:, None] for line in self.diagram.outgoing]
        decay_masses = []
        for layer_counts, layer_invariants in zip(
            reversed(self.diagram.s_decay_layers), reversed(self.s_decay_invariants)
        ):
            sqrt_s_min = []
            sqrt_s_index = 0
            layer_masses = []
            for decay_count in layer_counts:
                sqrt_s_min.append(torch.clip(sum(
                    sqrt_s[sqrt_s_index + i] for i in range(decay_count)
                ), min=self.sqrt_s_epsilon))
                layer_masses.append(sqrt_s[sqrt_s_index : sqrt_s_index + decay_count])
                sqrt_s_index += decay_count
            decay_masses.append(layer_masses)

            if len(layer_invariants) == 0:
                assert not self.has_t_channel
                continue

            sqrt_s = []
            invariant_iter = iter(layer_invariants)
            for i, decay_count in enumerate(layer_counts):
                if decay_count == 1:
                    sqrt_s.append(sqrt_s_min[i])
                    continue
                s_min = sqrt_s_min[i] ** 2
                s_max = (sqrt_s_hat[:, None] - sum(sqrt_s) - sum(sqrt_s_min[i+1:])) ** 2
                (s, ), jac = next(invariant_iter).map([rand()], condition=[s_min, s_max])
                sqrt_s.append(s.sqrt())
                ps_weight *= jac

        if self.has_t_channel:
            (p_in, p_out,), jac = self.t_mapping.map([
                rand(self.t_mapping.n_random), sqrt_s_hat, torch.cat(sqrt_s, dim=1)
            ])
            ps_weight *= jac
            p_out = p_out.unbind(dim=1)
        else:
            p_in = build_p_in(sqrt_s_hat)
            p_out = [s_hat]

        # build the momenta of the decays
        for layer_counts, layer_decays, layer_masses in zip(
            self.diagram.s_decay_layers, self.s_decays, reversed(decay_masses)
        ):
            p_out_prev = p_out
            p_out = []
            decay_iter = iter(layer_decays)
            for count, k_in, masses in zip(layer_counts, p_out_prev, layer_masses):
                if count == 1:
                    p_out.append(k_in)
                    continue
                m_out = torch.cat(masses, dim=1)
                (k_out, ), jac = next(decay_iter).map([rand(2), k_in, m_out])
                if len(k_in.shape) == 1:
                    mask = k_out.isnan().any(dim=1).any(dim=1)
                p_out.extend(k_out.unbind(dim=1))
                ps_weight *= jac
        p_out = torch.stack(p_out, dim=1)

        # we should have consumed all the random numbers
        assert rand.empty()

        # permute and return momenta
        p_ext = torch.cat([p_in, p_out[:, self.diagram.permutation]], dim=1)
        p_ext_lab = p_ext if self.luminosity is None else boost_beam(p_ext, rap)
        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        raise NotImplementedError("keine lust...")
