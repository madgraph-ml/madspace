from __future__ import annotations
from dataclasses import dataclass, field
from math import pi

import torch

from .base import PhaseSpaceMapping, TensorList
from .helper import boost_beam, lsquare
from .twoparticle import (
    tInvariantTwoParticleCOM,
    tInvariantTwoParticleLAB,
    TwoParticleLAB,
)
from .luminosity import Luminosity
from .invariants import BreitWignerInvariantBlock, UniformInvariantBlock

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


class DiagramMapping(PhaseSpaceMapping):
    """
    TODO:
        - support quartic vertices
        - leptonic initial state
        - pure s-channel diagrams
        - alternative strategy: chili + s-channel
        - alternative strategy: rambo + s-channel ?
    """
    def __init__(self, diagram: Diagram, s_lab: Tensor):
        n_out = len(diagram.outgoing)
        dims_in = [(3 * n_out - 2, )]
        dims_out = [(n_out, 4), (2, )]
        super().__init__(dims_in, dims_out)

        self.diagram = diagram

        self.s_lab = s_lab
        s_hat_min = torch.tensor([sum(line.mass for line in diagram.outgoing) ** 2])

        self.luminosity = Luminosity(s_lab, s_hat_min)

        last_t_line = diagram.t_channel_lines[-1]
        self.t_invariants = [
            tInvariantTwoParticleCOM(nu=1.4)
            if last_t_line.mass == 0. else
            tInvariantTwoParticleCOM(mass=last_t_line.mass, width=last_t_line.width)
        ]
        self.s_uniform_invariants = []
        for line in reversed(diagram.t_channel_lines[:-1]):
            self.t_invariants.append(
                tInvariantTwoParticleLAB(nu=1.4)
                if line.mass == 0. else
                tInvariantTwoParticleLAB(mass=line.mass, width=line.width)
            )
            self.s_uniform_invariants.append(UniformInvariantBlock())

        self.s_decay_invariants = []
        self.s_decays = []
        line_iter = iter(diagram.s_channel_lines)
        for layer in diagram.s_decay_layers:
            layer_invariants = []
            layer_decays = []
            for count in layer:
                if count == 1:
                    continue
                line = next(line_iter)
                layer_invariants.append(
                    BreitWignerInvariantBlock(mass=line.mass, width=line.width)
                    if line.mass != 0. else
                    (
                        MasslessInvariantBlock(width=line.width)
                        if line.width == 0. else
                        StableInvariantBlock()
                    )
                )
                layer_decays.append(TwoParticleLAB())
            self.s_decay_invariants.append(layer_invariants)
            self.s_decays.append(layer_decays)

        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        random = inputs[0]
        random_index = 0
        def rand(count=1):
            nonlocal random_index
            r = random[:, random_index : random_index + count]
            random_index += count
            return r
        ps_weight = 1.

        # Do luminosity and get s_hat and rapidity
        (x1x2,), jac_lumi = self.luminosity.map([rand(2)])
        ps_weight *= jac_lumi
        s_hat = self.s_lab * x1x2.prod(dim=1)
        sqrt_s_hat = s_hat.sqrt()
        rap = 0.5 * torch.log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # construct initial state momenta
        zeros = torch.zeros_like(sqrt_s_hat)
        p_cms = sqrt_s_hat / 2
        p1 = torch.stack([p_cms, zeros, zeros, p_cms], dim=1)
        p2 = torch.stack([p_cms, zeros, zeros, -p_cms], dim=1)
        p_in = torch.stack([p1, p2], dim=1)

        # sample s-invariants from decays, starting from the final state particles
        sqrt_s = [torch.full_like(sqrt_s_hat, line.mass) for line in self.diagram.outgoing]
        decay_masses = []
        for layer_counts, layer_invariants in zip(
            reversed(self.diagram.s_decay_layers), reversed(self.s_decay_invariants)
        ):
            sqrt_s_min = []
            sqrt_s_index = 0
            layer_masses = []
            for decay_count in layer_counts:
                sqrt_s_min.append(sum(
                    sqrt_s[sqrt_s_index + i] for i in range(decay_count)
                ))
                if decay_count > 1:
                    layer_masses.append(sqrt_s[sqrt_s_index : sqrt_s_index + decay_count])
                sqrt_s_index += decay_count
            decay_masses.append(layer_masses)

            sqrt_s = []
            for i, (decay_count, invariant) in enumerate(zip(layer_counts, layer_invariants)):
                if decay_count == 1:
                    sqrt_s.append(sqrt_s_min[i])
                    continue
                s_min = sqrt_s_min[i] ** 2
                s_max = (sqrt_s_hat - sum(sqrt_s) - sum(sqrt_s_min[i+1:])) ** 2
                (s, ), jac = invariant.map([rand(), m], condition=[s_min, s_max])
                sqrt_s.append(s.sqrt())
                ps_weight *= jac

        # sample s-invariants from the t-channel part of the diagram
        sqrt_s_min = sqrt_s[0][:, None]
        cumulated_sqrt_s = [sqrt_s_min]
        for invariant in self.s_uniform_invariants:
            s_min = sqrt_s_min ** 2
            s_max = (sqrt_s_hat[:, None] - self.diagram.outgoing[-1].mass) ** 2
            (s, ), jac = invariant.map([rand()], condition=[s_min, s_max])
            sqrt_s_min = s.sqrt()
            cumulated_sqrt_s.append(sqrt_s_min)
            ps_weight *= jac[:, 0]

        # sample t-invariants and build momenta of t-channel part of the diagram
        k_t = []
        p_t_in = p_in
        p2_rest = p2
        for invariant, cum_sqrt_s, out_sqrt_s in zip(
            self.t_invariants, reversed(cumulated_sqrt_s), reversed(sqrt_s[1:])
        ):
            m_t = torch.cat([cum_sqrt_s, out_sqrt_s[:,None]], dim=1)
            #print(m_t, cum_sqrt_s, out_sqrt_s, p_t_in)
            (ks, ), jac = invariant.map([rand(2), m_t], condition=[p_t_in])
            k_rest, k = ks[:, 0], ks[:, 1]
            k_t.append(k)
            p2_rest = p2_rest - k
            p_t_in = torch.stack([p1, p2_rest], dim=1)
            ps_weight *= jac
        k_t.append(k_rest)

        # build the momenta of the decays
        p_out_prev = torch.stack(list(reversed(k_t)), dim=1)
        for layer_counts, layer_decays, layer_masses in zip(
            self.diagram.s_decay_layers, self.s_decays, decay_masses
        ):
            p_out = []
            decay_iter = iter(layer_decays)
            for count, k_in, masses in zip(layer_counts, p_out, layer_masses):
                if count == 1:
                    p_out.append(k_in)
                    continue
                m_out = torch.stack(masses, dim=1)
                (k_out, ), jac = next(decay_iter).map([rand(2), k_in, m_out])
                p_out.append(k_out)
                ps_weight *= jac
            p_out_prev = torch.cat(p_out, dim=1)

        # we should have consumed all the random numbers
        assert random_index == random.shape[1]

        # permute and return momenta
        p_ext = torch.cat([p_in, p_out_prev[:, self.diagram.permutation]], dim=1)
        p_ext_lab = boost_beam(p_ext, rap)
        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        raise NotImplementedError("keine lust...")
