from .ir import IRFunction, IRType, IRVariable, scalar


VarList = list[IRVariable]
MapReturn = tuple[VarList, IRVariable]


class PhaseSpaceMapping:
    def map(self, ir: IRFunction, inputs: VarList, condition: VarList = []) -> MapReturn:
        self._check_types(inputs, self.types_in, "input")
        self._check_types(condition, self.types_c, "condition")
        outputs, det = self._map(ir, inputs, condition)
        self._check_types(outputs, self.types_out, "output")
        self._check_types([det], [scalar])
        return outputs, det

    def map_inverse(
        self, ir: IRFunction, inputs: VarList, condition: VarList = []
    ) -> MapReturn:
        self._check_types(inputs, self.types_in, "input")
        self._check_types(condition, self.types_c, "condition")
        outputs, det = self._map_inverse(ir, inputs, condition)
        self._check_types(outputs, self.types_out, "output")
        self._check_types([det], [scalar])
        return outputs, det

    def _map(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map(...) method"
        )

    def _map_inverse(self, ir: IRFunction, inputs: VarList, condition: VarList) -> MapReturn:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map_inverse(...) method"
        )

    def _check_types(
        self, variables: VarList, expected_types: list[IRType], name: str | None = None
    ):
        if len(variables) != len(expected_types):
            raise ValueError(
                f"wrong number of {name}s: expected {len(expected_types)}, got {len(variables)}"
            )
        for i, (var, type_expected) in enumerate(zip(variables, expected_types)):
            if var.type != type_expected:
                error = f"expected {type_expected}, got {var.type}"
                raise ValueError(error if name is None else f"{name} {i}: {error}")
