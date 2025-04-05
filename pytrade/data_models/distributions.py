import numpy as np

from typing import Any
from scipy.stats._distn_infrastructure import rv_continuous_frozen as RVContinuousFrozen


class TransformedDistribution:
    def __init__(self, generator: RVContinuousFrozen, transformer: Any | None = None):
        self._generator = generator
        self._transformer = transformer

    def rvs(self, size: int = 1):
        x = self._generator.rvs(size=size)
        if self._transformer is not None:
            if hasattr(self._transformer, "inverse_transform"):
                x = self._transformer.inverse_transform(x)
            else:
                raise RuntimeError("Transformer must have inverse_transform attribute")

        return x


class MixtureDistribution:
    """
    A class for a mixture of distribution.
    The distribution is simply given by
        f(x) = p*f1(x) + (1-p)*f2(x)

    where p is the mixture parameter in [0, 1] and
    each distribution can assume different parameters.

    :ivar mixture_param is the share in [0, 1] that controls the mixture
    :ivar generators is a list that contains the different distribution generators
    """

    def __init__(self, mixture_param: float, generators: list[RVContinuousFrozen]):

        self._mixture_param = mixture_param
        self._generators = generators

    def rvs(self, size=1):
        boolean_mask = np.random.binomial(n=1, p=self._mixture_param, size=size)
        x = boolean_mask * self._generators[0].rvs(size) - (
            1 - boolean_mask
        ) * self._generators[1].rvs(size)
        return x


class DiscreteConditionalDistribution:

    def __init__(
        self, generators: list[RVContinuousFrozen], conditional_bins: list[float]
    ):
        self._generators = generators
        self._conditional_bins = conditional_bins

    def rvs(self, conditional):

        size = len(conditional)

        # Generate boolean mask
        boolean_mask = np.zeros((len(self._conditional_bins) - 1, size), dtype=int)
        for i, (lower, upper) in enumerate(
            zip(self._conditional_bins[:-1], self._conditional_bins[1:])
        ):
            boolean_mask[i, :] = (conditional >= lower) & (conditional < upper)

        boolean_mask = np.argmax(
            boolean_mask, axis=0
        )  # Converts to 1d vector with correct entry

        # Generate random draws from generators
        x = np.zeros((size, len(self._generators)))
        for i in range(len(self._generators)):
            x[:, i] = self._generators[i].rvs(size)

        x_c = np.zeros(size)
        for i, idx in enumerate(boolean_mask):
            x_c[i] = x[i, idx]

        return x_c
