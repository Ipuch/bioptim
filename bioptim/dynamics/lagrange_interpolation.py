from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from casadi import collocation_points, MX, SX

from typing import Any, Callable as CallableObj
import numpy as np
from typing import TypeAlias
from casadi import MX, SX, DM


Int: TypeAlias = int
Range: TypeAlias = range
Str: TypeAlias = str
Float: TypeAlias = float
Bool: TypeAlias = bool
Tuple: TypeAlias = tuple
List: TypeAlias = list
Bytes: TypeAlias = bytes
Callable: TypeAlias = CallableObj

IntorFloat: TypeAlias = int | float

AnyIterable: TypeAlias = list[Any] | tuple[Any, ...]
AnyIterableOptional: TypeAlias = list[Any] | tuple[Any, ...] | None
Indexer: TypeAlias = slice | list | range
IndexerOptional: TypeAlias = slice | list | range | None
AnyIterableOrSlice: TypeAlias = slice | list | tuple
AnyIterableOrSliceOptional: TypeAlias = slice | list | tuple | None
AnySequence: TypeAlias = list | tuple | range | np.ndarray
AnySequenceOptional: TypeAlias = list | tuple | range | np.ndarray | None
IntIterable: TypeAlias = list[int] | tuple[int, ...]
IntIterableOptional: TypeAlias = list[int] | tuple[int, ...] | None
StrIterable: TypeAlias = list[str] | tuple[str, ...]

StrIterableOptional: TypeAlias = list[str] | tuple[str, ...] | None

AnyDict: TypeAlias = dict[str, Any]
AnyListorDict: TypeAlias = list[Any] | dict[str, Any]
IntDict: TypeAlias = dict[str, int]
NpArrayDict: TypeAlias = dict[str, np.ndarray]
NpArrayDictOptional: TypeAlias = NpArrayDict | None

AnyDictOptional: TypeAlias = dict[str, Any] | None
AnyListOptional: TypeAlias = list[Any] | None

AnyList: TypeAlias = list[Any]
BoolList: TypeAlias = list[bool]
IntList: TypeAlias = list[int]
FloatList: TypeAlias = list[float]
StrList: TypeAlias = list[str]
MXList: TypeAlias = list[MX]
DMList: TypeAlias = list[DM]
NpArrayList: TypeAlias = list[np.ndarray]
NpArray: TypeAlias = np.ndarray

NpArrayorFloat: TypeAlias = np.ndarray | float
NpArrayorFloatOptional: TypeAlias = np.ndarray | float | None
FloatIterableorNpArray: TypeAlias = list[float] | tuple[float, ...] | np.ndarray
FloatIterableorNpArrayorFloat: TypeAlias = FloatIterableorNpArray | float
IntIterableorNpArray: TypeAlias = list[int] | tuple[int, ...] | range | np.ndarray
IntIterableorNpArrayorInt: TypeAlias = int | IntIterableorNpArray

IntListOptional: TypeAlias = list[int] | None
FloatListOptional: TypeAlias = list[float] | None
StrListOptional: TypeAlias = list[str] | None
NpArrayListOptional: TypeAlias = list[np.ndarray] | None

StrOrIterable: TypeAlias = str | list[str]

IntOptional: TypeAlias = int | None
IntOrStr: TypeAlias = int | str
IntOrStrOptional: TypeAlias = int | str | None

FloatOptional: TypeAlias = float | None

StrOptional: TypeAlias = str | None

BoolOptional: TypeAlias = bool | None

NpArrayOptional: TypeAlias = np.ndarray | None

AnyTuple: TypeAlias = tuple[Any, ...]
IntTuple: TypeAlias = tuple[int, ...]
FloatTuple: TypeAlias = tuple[float, ...]
StrTuple: TypeAlias = tuple[str, ...]
DoubleIntTuple: TypeAlias = tuple[int, int]
DoubleFloatTuple: TypeAlias = tuple[float, float]
DoubleNpArrayTuple: TypeAlias = tuple[np.ndarray, np.ndarray]


IntStrorIterable: TypeAlias = int | str | AnyIterable
IntorIterableOptional: TypeAlias = IntOptional | IntList | IntTuple

CX: TypeAlias = MX | SX
CXOptional: TypeAlias = MX | SX | None
CXorFloat: TypeAlias = CX | float
CXorDM: TypeAlias = MX | SX | DM | float
CXorDMorFloatIterable: TypeAlias = FloatIterableorNpArray | MX | SX | DM
CXorDMorNpArray: TypeAlias = np.ndarray | MX | SX | DM
CXList: TypeAlias = list[CX]


class LagrangeInterpolation:
    def __init__(
        self,
        time_grid: FloatList,
    ):
        self.time_grid = time_grid

    @cached_property
    def polynomial_degree(self) -> Int:
        return len(self.time_grid)

    def lagrange_polynomial(self, j: Int, time_control_interval: CX) -> CX:
        """
        Compute the j-th Lagrange polynomial \\(L_j(\tau)\\) in symbolic form.

        This polynomial is defined so that:

        .. math::
            L_j(\tau_i) = \\delta_{ij} =
            \\begin{cases}
                1 & \\text{if } i = j, \\\\
                0 & \\text{if } i \\neq j
            \\end{cases}

        for each of the collocation (or interpolation) points \\(\\tau_i\\) given in
        ``integration_time``. In other words, \\(L_j(\\tau)\\) is a polynomial in
        \\(\\tau\\) that is 1 at \\(\\tau_j = \\text{integration_time}[j]\\) and 0
        at every other \\(\\tau_r\\) with \\(r \\neq j\\).

        Parameters
        ----------
        j : int
            Index (0-based) of the Lagrange polynomial to compute,
            corresponding to the j-th collocation point.
        time_control_interval : MX or SX
            The symbolic variable (CasADi `MX` or `SX`) representing
            the dimensionless time within the control interval.

        Returns
        -------
        MX or SX
            A CasADi symbolic expression representing the j-th Lagrange polynomial
            evaluated at ``time_control_interval``.

        Notes
        -----
        The Lagrange polynomial \\( L_j(\\tau) \\) is constructed by the product:

        math::
            L_j(\\tau)
            = \\prod_{\\substack{r=0 \\\\ r \\neq j}}^{\\text{polynomial_order}-1}
              \\frac{\\tau - \\tau_r}{\\tau_j - \\tau_r},

        where \\(\\tau_j = \\text{integration_time}[j]\\) and
        \\(\\tau_r = \\text{integration_time}[r]\\). This function thus
        returns the polynomial in a symbolic form that can be used for
        further manipulation or evaluation (e.g., computing derivatives).
        """
        _l = 1
        for r in range(self.polynomial_degree):
            if r != j:
                _l *= (time_control_interval - self.time_grid[r]) / (self.time_grid[j] - self.time_grid[r])
        return _l

    def partial_lagrange_polynomial(self, j: Int, time_control_interval: CX, i: Int) -> CX:
        """Compute the partial product of the j-th Lagrange polynomial without the i-th term."""
        _l = 1
        for r in range(self.polynomial_degree):
            if r != j and r != i:
                _l *= (time_control_interval - self.time_grid[r]) / (self.time_grid[j] - self.time_grid[r])
        return _l

    def lagrange_polynomial_derivative(self, j: Int, time_control_interval: CX) -> CX:
        """
        Compute the derivative of the j-th Lagrange polynomial \\( L_j(\\tau) \\) in symbolic form.

        By factoring out \\( L_j(\\tau) \\) and removing one factor from the product,
        the derivative can be expressed as:

        .. math::
            \\frac{d}{d\\tau} L_j(\\tau)
            = L_j(\\tau)
              \\sum_{\\substack{k=0 \\\\ k \\neq j}}^{\\text{polynomial_order}-1}
                \\frac{1}{\\tau - \\tau_k}.

        Parameters
        ----------
        j : int
            Index (0-based) of the Lagrange polynomial whose derivative is computed.
        time_control_interval : MX or SX
            The symbolic variable (CasADi `MX` or `SX`) representing
            the dimensionless time within the control interval.

        Returns
        -------
        MX or SX
            A CasADi symbolic expression representing the derivative
            of the j-th Lagrange polynomial evaluated at ``time_control_interval``.

        Notes
        -----
        The full expansion can also be written as:

        .. math::
            \\frac{d}{d\\tau} L_j(\\tau)
            = \\sum_{k \\neq j} \\left[
                \\frac{1}{(\\tau_j - \\tau_k)}
                \\prod_{\\substack{r \\neq j, r \\neq k}}^{} \\frac{(\\tau - \\tau_r)}{(\\tau_j - \\tau_r)}
              \\right],

        which, when factored appropriately, yields the compact form above.
        """
        sum_term = 0
        for k in range(self.polynomial_degree):
            if k == j:
                continue

            partial_Ljk = self.partial_lagrange_polynomial(j, time_control_interval, k)
            sum_term += 1.0 / (self.time_grid[j] - self.time_grid[k]) * partial_Ljk

        return sum_term

    def interpolate(self, y_values: list[CXorFloat], time_control_interval: CXorFloat) -> CX:
        """
        Compute the Lagrange interpolation of a set of values at the given time.

        Parameters
        ----------
        y_values : list of MX or SX
            The values to interpolate at each collocation point.
        time_control_interval : MX or SX
            The symbolic variable (CasADi `MX` or `SX`) representing
            the dimensionless time within the control interval.

        Returns
        -------
        MX or SX
            A CasADi symbolic expression representing the Lagrange interpolation
            of the given values at the time ``time_control_interval``.
        """
        self._check_y_values(y_values)
        interpolated_value = 0
        for j in range(self.polynomial_degree):
            interpolated_value += y_values[j] * self.lagrange_polynomial(j, time_control_interval)
        return interpolated_value

    def interpolate_first_derivative(self, y_values: list[CXorFloat], time_control_interval: CXorFloat) -> CX:
        """
        Compute the first derivative of the Lagrange interpolation of a set of values at the given time.

        Parameters
        ----------
        y_values : list of MX or SX
            The values to interpolate at each collocation point.
        time_control_interval : MX or SX
            The symbolic variable (CasADi `MX` or `SX`) representing
            the dimensionless time within the control interval.

        Returns
        -------
        MX or SX
            A CasADi symbolic expression representing the first derivative
            of the Lagrange interpolation of the given values at the time ``time_control_interval``.
        """
        self._check_y_values(y_values)
        interpolated_value = 0
        for j in range(self.polynomial_degree):
            interpolated_value += y_values[j] * self.lagrange_polynomial_derivative(j, time_control_interval)
        return interpolated_value

    def _check_y_values(self, y_values) -> None:
        if len(y_values) != self.polynomial_degree:
            raise ValueError(
                f"Length of y_values ({len(y_values)}) must match the polynomial order ({self.polynomial_degree})"
            )


def main():
    # Choose polynomial_order and get collocation points
    polynomial_order = 3
    time_grid = [0] + collocation_points(polynomial_order, "legendre")

    # Some example y-values
    y_dict = {
        1: [0, 2],
        2: [0, 0.3, 2],
        3: [0, 0.15, -0.45, 2],
        4: [0, 0.1, -0.2, 0.7, 2],
        5: [0, 0.075, 0.15, -0.4, 0.95, 2],
    }
    y_values = y_dict[polynomial_order]

    # A fine grid for plotting
    continuous_time = np.linspace(0, 1, 1000)

    # Create the Lagrange interpolation object
    lagrange_interp = LagrangeInterpolation(time_grid)

    # Interpolate the y-values at the fine grid
    interpolated_values = [lagrange_interp.interpolate(y_values, t) for t in continuous_time]
    interpolated_velocity = [lagrange_interp.interpolate_first_derivative(y_values, t) for t in continuous_time]

    # Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(continuous_time, interpolated_values, label="Interpolated Value")
    ax[0].plot(time_grid, y_values, "o", label="Collocation Points")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(True)

    for i in range(polynomial_order + 1):
        ax[0].plot(
            continuous_time,
            [lagrange_interp.lagrange_polynomial(i, t) * y_values[i] for t in continuous_time],
            label=f"L_{i}(tau)",
            alpha=0.3,
        )

    ax[1].plot(continuous_time, interpolated_velocity, label="Interpolated Velocity")

    for i in range(polynomial_order + 1):
        ax[1].plot(
            continuous_time,
            [lagrange_interp.lagrange_polynomial_derivative(i, t) * y_values[i] for t in continuous_time],
            label=f"dL_{i}/dtau",
            alpha=0.3,
        )

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Velocity")
    ax[1].legend()
    ax[1].grid(True)

    plt.show()


if __name__ == "__main__":
    main()
