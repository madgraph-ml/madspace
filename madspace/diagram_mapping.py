from __future__ import annotations
from dataclasses import dataclass, field

import torch

from .base import PhaseSpaceMapping, TensorList

@dataclass
class Line:
    mass: float = 0.
    width: float = 0.
    name: str | None = None
    vertices: list[Vertex] = field(init=False)

    def __post_init__(self):
        self.vertices = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.name is not None:
            return self.name
        elif len(self.vertices) == 2:
            return f"{self.vertices[0]} -- {self.vertices[1]}"
        else:
            return "?"

@dataclass
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
    s_channel_vertices: list[Vertex] = field(init=False)
    s_channel_lines: list[Line] = field(init=False)

    def __post_init__(self):
        self._fill_names(self.vertices, "v")
        self._fill_names(self.incoming, "in")
        self._fill_names(self.outgoing, "out")

        (
            t_channel_lines, self.t_channel_vertices
        ) = self._t_channel_recursive(self.incoming[0], None)
        self.t_channel_lines = t_channel_lines[1:]
        self._fill_names(self.t_channel_lines, "t")

        self.s_channel_lines, self.s_channel_vertices = self._s_channel()
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

    def _s_channel(self) -> list[Line]:
        t_channel_lines = [self.incoming[0], *self.t_channel_lines, self.incoming[1]]
        lines = []
        vertices = []
        for vertex, line_in_1, line_in_2 in zip(
            self.t_channel_vertices, t_channel_lines[:-1], t_channel_lines[1:]
        ):
            for line in vertex.lines:
                if line in [line_in_1, line_in_2]:
                    continue
                lines.append(line)
                vertices.append(vertex)

        s_channel_lines = []
        s_channel_vertices = []
        while len(lines) != 0:
            next_lines = []
            next_vertices = []
            for line, parent_vertex in zip(lines, vertices):
                if line in self.outgoing:
                    continue
                s_channel_lines.append(line)
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                s_channel_vertices.append(vertex)

                for next_line in vertex.lines:
                    if next_line is line:
                        continue
                    next_lines.append(next_line)
                    next_vertices.append(vertex)
            lines = next_lines
            vertices = next_vertices

        return s_channel_lines, s_channel_vertices


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

        self.s_lab = s_lab
        s_hat_min = sum(line.mass for line in outgoing) ** 2

        self.luminosity = Luminosity(s_lab, s_hat_min)
        #self.t_invariants = [tInv
        #for line in diagram.t_channel_lines:

        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)
