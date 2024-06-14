from .ir import IRType, IRVariable

VarList = list[IRVariable]
MapReturn = tuple[VarList, IRVariable]


class PhaseSpaceMapping:
    def map(self, ir: IRFunction, inputs: VarList, condition: VarList = []) -> MapReturn:
        self._check_types(inputs, self.types_in, "input")
        self._check_types(condition, self.types_c, "condition")
        outputs, det = self._map(inputs, condition)
        self._check_types(outputs, self.types_out, "output")
        self._check_types([det], [scalar])
        return outputs, det

    def map_inverse(
        self, ir: IRFunction, inputs: VarList, condition: VarList = []
    ) -> MapReturn:
        self._check_types(inputs, self.types_in, "input")
        self._check_types(condition, self.types_c, "condition")
        outputs, det = self._map_inverse(inputs, condition)
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
        self, types: list[IRType], expected_types: list[IRType], name: str | None
    ):
        if len(types) != len(expected_types):
            raise ValueError(
                f"wrong number of {name}s: expected {len(expected_types)}, got {len(types)}"
            )
        for i, (type_given, type_expected) in enumerate(zip(types, expected_types)):
            if type_given != type_expected:
                error = f"expected {type_expected}, got {type_given}"
                raise ValueError(error if name is None else f"{name} {i}: {error}")
