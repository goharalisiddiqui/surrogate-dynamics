
import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Num


class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]: ...

    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)

    def __radd__(self, other):
        if other == 0:
            return self
        return other + self


@dataclass
class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    transformations: list[Transformation]

    def __post_init__(self) -> None:
        transformations = []

        for transformation in self.transformations:
            if isinstance(transformation, Identity):
                continue
            elif isinstance(transformation, Chain):
                transformations.extend(transformation.transformations)
            else:
                assert isinstance(transformation, Transformation)
                transformations.append(transformation)

        self.transformations = transformations
        self.__init_passed_kwargs__ = {"transformations": transformations}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        for t in self.transformations:
            data_entry = t(data_entry)
        return data_entry


class Identity(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        return data_entry


class ApplyFuncMixin:
    @staticmethod
    def apply_func(
        func: Callable[[dict[str, Any], str], None],
        data_entry: dict[str, Any],
        fields: tuple[str, ...],
        optional_fields: tuple[str, ...] = (),
    ):
        for field in fields:
            func(data_entry, field)
        for field in optional_fields:
            if field in data_entry:
                func(data_entry, field)


class ImputationMethod:
    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]: ...


@dataclass(frozen=True)
class DummyValueImputation(ImputationMethod):
    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]:
        x[np.isnan(x)] = self.value
        return x


@dataclass(frozen=True)
class LastValueImputation(ImputationMethod):
    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"]
    ) -> Num[np.ndarray, "length *dim"]:
        x = x.T
        x[0:1][np.isnan(x[0:1])] = self.value
        mask = np.isnan(x)
        idx = np.arange(len(x))
        if x.ndim == 2:
            idx = np.expand_dims(idx, axis=1)
        idx = np.where(~mask, idx, 0)
        idx = np.maximum.accumulate(idx, axis=0)
        if x.ndim == 2:
            x = x[idx, np.arange(x.shape[1])]
        else:
            x = x[idx]
        return x.T


@dataclass(frozen=True)
class CausalMeanImputation(ImputationMethod):
    """
    This class replaces each missing value with the average of all the values
    up to this point, ensuring causality.

    - If the first values are missing, they are replaced by the closest non-missing value.
    - If an entire sequence is NaN, it is replaced by a predefined value.
    """

    value: int | float | complex = 0.0

    def __call__(
        self, x: Num[np.ndarray, "length *dim"], value: int | float | complex = 0.0
    ) -> Num[np.ndarray, "length *dim"]:
        mask = np.isnan(x).T

        # do last value imputation first
        last_value_imputation = LastValueImputation(self.value)
        x = last_value_imputation(x)
        mask[0] = False
        x = x.T

        if x.ndim == 1:
            adjusted_values_to_causality = np.concatenate((np.repeat(0.0, 1), x[:-1]))
            cumsum = np.cumsum(adjusted_values_to_causality)
            indices = np.linspace(0, len(x) - 1, len(x))
            indices[0] = 1
            ar_res = cumsum / indices
            x[mask] = ar_res[mask]
        else:
            # compute cumulative sum
            adjusted_values_to_causality = np.vstack(
                (np.zeros((1, x.shape[1])), x[:-1, :])
            )
            cumsum = np.cumsum(adjusted_values_to_causality, axis=0)

            # compute causal mean
            indices = np.linspace(0, len(x) - 1, len(x)).reshape(-1, 1)
            indices[0] = 1
            ar_res = cumsum / indices
            # impute with causal mean
            x[mask] = ar_res[mask]
        return x.T


@dataclass
class ImputeTimeSeries(ApplyFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    imputation_method: ImputationMethod = DummyValueImputation(value=0.0)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.apply_func(
            self._impute,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _impute(self, data_entry: dict[str, Any], field: str):
        value = data_entry[field]
        nan_entries = np.isnan(value)
        if nan_entries.any():
            data_entry[field] = self.imputation_method(value)