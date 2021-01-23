from typing import Union, Callable

import numpy as np
from casadi import MX, SX, vertcat
from scipy.interpolate import interp1d

from ..misc.enums import InterpolationType
from ..misc.mapping import BidirectionalMapping
from ..misc.options import UniquePerPhaseOptionList, OptionGeneric


class PathCondition(np.ndarray):
    """
    A matrix for any component (rows) and time (columns) conditions

    Attributes
    ----------
    nb_shooting: int
        Number of shooting points
    type: InterpolationType
        Type of interpolation
    t: list[float]
        Time vector
    extra_params: dict
        Any extra parameters that is associated to the path condition
    slice_list: slice
        Slice of the array
    custom_function: function
        Custom function to describe the path condition interpolation

    Methods
    -------
    __array_finalize__(self, obj: "PathCondition")
        Finalize the array. This is required since PathCondition inherits from np.ndarray
    __reduce__(self) -> tuple
        Adding some attributes to the reduced state
    __setstate__(self, state: tuple, *args, **kwargs)
        Adding some attributes to the expanded state
    check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int, element_type: InterpolationType)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    evaluate_at(self, shooting_point: int)
        Evaluate the interpolation at a specific shooting point
    """

    def __new__(
        cls,
        input_array: Union[np.ndarray, Callable],
        t: list = None,
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        slice_list: Union[slice, list, tuple] = None,
        **extra_params,
    ):
        """
        Parameters
        ----------
        input_array: Union[np.ndarray, Callable]
            The matrix of interpolation, rows are the components, columns are the time
        t: list[float]
            The time stamps
        interpolation: InterpolationType
            The type of interpolation. It determines how many timestamps are required
        slice_list: Union[slice, list, tuple]
            If the data should be sliced. It is more relevant for custom functions
        extra_params: dict
            Any parameters to pass to the path condition
        """

        # Check and reinterpret input
        custom_function = None
        if interpolation == InterpolationType.CUSTOM:
            if not callable(input_array):
                raise TypeError("The input when using InterpolationType.CUSTOM should be a callable function")
            custom_function = input_array
            input_array = np.array(())
        if not isinstance(input_array, (MX, SX)):
            input_array = np.asarray(input_array, dtype=float)

        if len(input_array.shape) == 0:
            input_array = input_array[np.newaxis, np.newaxis]

        if interpolation == InterpolationType.CONSTANT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 1"
                )

        elif interpolation == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if len(input_array.shape) == 1:
                input_array = input_array[:, np.newaxis]
            if input_array.shape[1] != 1 and input_array.shape[1] != 3:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 1 or 3"
                )
            if input_array.shape[1] == 1:
                input_array = np.repeat(input_array, 3, axis=1)
        elif interpolation == InterpolationType.LINEAR:
            if input_array.shape[1] != 2:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.LINEAR_CONTINUOUS "
                    f"(ncols = {input_array.shape[1]}), the expected number of column is 2"
                )
        elif interpolation == InterpolationType.EACH_FRAME:
            # This will be verified when the expected number of columns is set
            pass
        elif interpolation == InterpolationType.SPLINE:
            if input_array.shape[1] < 2:
                raise RuntimeError("Value for InterpolationType.SPLINE must have at least 2 columns")
            if t is None:
                raise RuntimeError("Spline necessitate a time vector")
            t = np.asarray(t)
            if input_array.shape[1] != t.shape[0]:
                raise RuntimeError("Spline necessitate a time vector which as the same length as column of data")

        elif interpolation == InterpolationType.CUSTOM:
            # We have to assume dimensions are those the user wants
            pass
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")
        if not isinstance(input_array, (MX, SX)):
            obj = np.asarray(input_array).view(cls)
        else:
            obj = input_array

        # Additional information (do not forget to update __reduce__ and __setstate__)
        obj.nb_shooting = None
        obj.type = interpolation
        obj.t = t
        obj.extra_params = extra_params
        obj.slice_list = slice_list
        if interpolation == InterpolationType.CUSTOM:
            obj.custom_function = custom_function

        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array. This is required since PathCondition inherits from np.ndarray

        Parameters
        ----------
        obj: PathCondition
            The current object to finalize
        """

        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.nb_shooting = getattr(obj, "nb_shooting", None)
        self.type = getattr(obj, "type", None)
        self.t = getattr(obj, "t", None)
        self.extra_params = getattr(obj, "extra_params", None)
        self.slice_list = getattr(obj, "slice_list", None)

    def __reduce__(self) -> tuple:
        """
        Adding some attributes to the reduced state

        Returns
        -------
        The reduced state of the class
        """

        pickled_state = super(PathCondition, self).__reduce__()
        new_state = pickled_state[2] + (self.nb_shooting, self.type, self.t, self.extra_params, self.slice_list)
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state: tuple, *args, **kwargs):
        """
        Adding some attributes to the expanded state

        Parameters
        ----------
        state: tuple
            The state as described by __reduce__
        """

        self.nb_shooting = state[-5]
        self.type = state[-4]
        self.t = state[-3]
        self.extra_params = state[-2]
        self.slice_list = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super(PathCondition, self).__setstate__(state[0:-5], *args, **kwargs)

    def check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int, element_type: InterpolationType):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        nb_elements: int
            The expected number of rows
        nb_shooting: int
            The number of shooting points in the ocp
        element_type: InterpolationType
            The type of the interpolation
        """

        if (
            self.type == InterpolationType.CONSTANT
            or self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT
            or self.type == InterpolationType.LINEAR
            or self.type == InterpolationType.SPLINE
            or self.type == InterpolationType.CUSTOM
        ):
            self.nb_shooting = nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            self.nb_shooting = nb_shooting + 1
        else:
            if self.nb_shooting != nb_shooting:
                raise RuntimeError(
                    f"Invalid number of shooting ({self.nb_shooting}), the expected number is {nb_shooting}"
                )

        if self.type == InterpolationType.CUSTOM:
            slice_list = self.slice_list
            if slice_list is not None:
                val_size = self.custom_function(0, **self.extra_params)[
                    slice_list.start : slice_list.stop : slice_list.step
                ].shape[0]
            else:
                val_size = self.custom_function(0, **self.extra_params).shape[0]
        else:
            val_size = self.shape[0]
        if val_size != nb_elements:
            raise RuntimeError(f"Invalid number of {element_type} ({val_size}), the expected size is {nb_elements}")

        if self.type == InterpolationType.EACH_FRAME:
            if self.shape[1] != self.nb_shooting:
                raise RuntimeError(
                    f"Invalid number of column for InterpolationType.EACH_FRAME (ncols = {self.shape[1]}), "
                    f"the expected number of column is {self.nb_shooting}"
                )

    def evaluate_at(self, shooting_point: int):
        """
        Evaluate the interpolation at a specific shooting point

        Parameters
        ----------
        shooting_point: int
            The shooting point to evaluate the path condition at

        Returns
        -------
        The values of the components at a specific time index
        """

        if self.nb_shooting is None:
            raise RuntimeError(f"check_and_adjust_dimensions must be called at least once before evaluating at")

        if self.type == InterpolationType.CONSTANT:
            return self[:, 0]
        elif self.type == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
            if shooting_point == 0:
                return self[:, 0]
            elif shooting_point == self.nb_shooting:
                return self[:, 2]
            elif shooting_point > self.nb_shooting:
                raise RuntimeError("shooting point too high")
            else:
                return self[:, 1]
        elif self.type == InterpolationType.LINEAR:
            return self[:, 0] + (self[:, 1] - self[:, 0]) * shooting_point / self.nb_shooting
        elif self.type == InterpolationType.EACH_FRAME:
            return self[:, shooting_point]
        elif self.type == InterpolationType.SPLINE:
            spline = interp1d(self.t, self)
            return spline(shooting_point / self.nb_shooting * (self.t[-1] - self.t[0]))
        elif self.type == InterpolationType.CUSTOM:
            if self.slice_list is not None:
                slice_list = self.slice_list
                return self.custom_function(shooting_point, **self.extra_params)[
                    slice_list.start : slice_list.stop : slice_list.step
                ]
            else:
                return self.custom_function(shooting_point, **self.extra_params)
        else:
            raise RuntimeError(f"InterpolationType is not implemented yet")


class Bounds(OptionGeneric):
    """
    A placeholder for bounds constraints

    Attributes
    ----------
    nb_shooting: int
        The number of shooting of the ocp
    min: PathCondition
        The minimal bound
    max: PathCondition
        The maximal bound
    type: InterpolationType
        The type of interpolation of the bound
    t: list[float]
        The time stamps
    extra_params: dict
        Any parameters to pass to the path condition

    Methods
    -------
    check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    concatenate(self, other: "Bounds")
        Vertical concatenate of two Bounds
    __getitem__(self, slice_list: slice) -> "Bounds"
        Allows to get from square brackets
    __setitem__(self, slice: slice, value: Union[np.ndarray, float])
        Allows to set from square brackets
    __bool__(self) -> bool
        Get if the Bounds is empty
    shape(self) -> int
        Get the size of the Bounds
    """

    def __init__(
        self,
        min_bound: Union[PathCondition, np.ndarray, list, tuple] = (),
        max_bound: Union[PathCondition, np.ndarray, list, tuple] = (),
        interpolation: InterpolationType = InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        slice_list: Union[slice, list, tuple] = None,
        **parameters,
    ):
        """
        Parameters
        ----------
        min_bound: Union[PathCondition, np.ndarray, list, tuple]
            The minimal bound
        max_bound: Union[PathCondition, np.ndarray, list, tuple]
            The maximal bound
        interpolation: InterpolationType
            The type of interpolation of the bound
        slice_list: Union[slice, list, tuple]
            Slice of the array
        parameters: dict
            Any extra parameters that is associated to the path condition
        """
        if isinstance(min_bound, PathCondition):
            self.min = min_bound
        else:
            self.min = PathCondition(min_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        if isinstance(max_bound, PathCondition):
            self.max = max_bound
        else:
            self.max = PathCondition(max_bound, interpolation=interpolation, slice_list=slice_list, **parameters)

        super(Bounds, self).__init__(**parameters)
        self.type = interpolation
        self.t = None
        self.extra_params = self.min.extra_params
        self.nb_shooting = self.min.nb_shooting

    def check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        nb_elements: int
            The expected number of rows
        nb_shooting: int
            The number of shooting points in the ocp
        """

        self.min.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound min")
        self.max.check_and_adjust_dimensions(nb_elements, nb_shooting, "Bound max")
        self.t = self.min.t
        self.nb_shooting = self.min.nb_shooting

    def concatenate(self, other: "Bounds"):
        """
        Vertical concatenate of two Bounds

        Parameters
        ----------
        other: Bounds
            The Bounds to concatenate with
        """

        if not isinstance(self.min, (MX, SX)) and not isinstance(other.min, (MX, SX)):
            self.min = PathCondition(np.concatenate((self.min, other.min)), interpolation=self.min.type)
        else:
            self.min = PathCondition(vertcat(self.min, other.min), interpolation=self.min.type)
        if not isinstance(self.max, (MX, SX)) and not isinstance(other.max, (MX, SX)):
            self.max = PathCondition(np.concatenate((self.max, other.max)), interpolation=self.max.type)
        else:
            self.max = PathCondition(vertcat(self.max, other.max), interpolation=self.max.type)

        self.type = self.min.type
        self.t = self.min.t
        self.extra_params = self.min.extra_params
        self.nb_shooting = self.min.nb_shooting

    def __getitem__(self, slice_list: Union[slice, list, tuple]) -> "Bounds":
        """
        Allows to get from square brackets

        Parameters
        ----------
        slice_list: Union[slice, list, tuple]
            The slice to get

        Returns
        -------
        The bound sliced
        """

        if isinstance(slice_list, slice):
            t = self.min.t
            param = self.extra_params
            interpolation = self.type
            if interpolation == InterpolationType.CUSTOM:
                min_bound = self.min.custom_function
                max_bound = self.max.custom_function
            else:
                min_bound = np.array(self.min[slice_list.start : slice_list.stop : slice_list.step])
                max_bound = np.array(self.max[slice_list.start : slice_list.stop : slice_list.step])
            bounds_sliced = Bounds(
                min_bound=min_bound,
                max_bound=max_bound,
                interpolation=interpolation,
                slice_list=slice_list,
                t=t,
                **param,
            )
            # TODO: Verify if it is ok that slice_list arg sent is used only if it is a custom type
            #  (otherwise, slice_list is used before calling Bounds constructor)
            return bounds_sliced
        else:
            raise RuntimeError(
                "Invalid input for slicing bounds. Please note that columns should not be specified. "
                "Therefore, it should look like [a:b] or [a:b:c] where a is the starting index, "
                "b is the stopping index and c is the step for slicing."
            )

    def __setitem__(self, _slice: Union[slice, list, tuple], value: Union[np.ndarray, float]):
        """
        Allows to set from square brackets

        Parameters
        ----------
        _slice: Union[slice, list, tuple]
            The slice where to put the data
        value: Union[np.ndarray, float]
            The value to set
        """

        self.min[_slice] = value
        self.max[_slice] = value

    def __bool__(self) -> bool:
        """
        Get if the Bounds is empty

        Returns
        -------
        If the Bounds is empty
        """

        return len(self.min) > 0

    @property
    def shape(self) -> int:
        """
        Get the size of the Bounds

        Returns
        -------
        The size of the Bounds
        """

        return self.min.shape


class BoundsList(UniquePerPhaseOptionList):
    """
    A list of Bounds if more than one is required

    Methods
    -------
    add(self, min_bound: Union[PathCondition, np.ndarray] = None, max_bound: Union[PathCondition, np.ndarray] = None,
            bounds: Bounds = None, **extra_arguments)
        Add a new constraint to the list, either [min_bound AND max_bound] OR [bounds] should be defined
    __getitem__(self, i: int)
        Get the ith bounds of the list
    __next__(self)
        Get the next bounds of the list
    print(self)
        Print the BoundsList to the console
    """

    def add(
        self,
        min_bound: Union[PathCondition, np.ndarray] = None,
        max_bound: Union[PathCondition, np.ndarray] = None,
        bounds: Bounds = None,
        **extra_arguments,
    ):
        """
        Add a new bounds to the list, either [min_bound AND max_bound] OR [bounds] should be defined

        Parameters
        ----------
        min_bound: Union[PathCondition, np.ndarray]
            The minimum path condition. If min_bound if defined, then max_bound must be so and bound should be None
        max_bound: [PathCondition, np.ndarray]
            The maximum path condition. If max_bound if defined, then min_bound must be so and bound should be None
        bounds: Bounds
            Copy a Bounds. If bounds is defined, min_bound and max_bound should be None
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if bounds and (min_bound or max_bound):
            RuntimeError("min_bound/max_bound and bounds cannot be set alongside")
        if isinstance(bounds, Bounds):
            if bounds.phase == -1:
                bounds.phase = len(self.options) if self.options[0] else 0
            self.copy(bounds)
        else:
            super(BoundsList, self)._add(
                min_bound=min_bound, max_bound=max_bound, option_type=Bounds, **extra_arguments
            )

    def __getitem__(self, i: int):
        """
        Get the ith bounds of the list

        Parameters
        ----------
        i: int
            The index of the bounds to get

        Returns
        -------
        The ith bounds of the list
        """

        return super(BoundsList, self).__getitem__(i)

    def __next__(self):
        """
        Get the next bounds of the list

        Returns
        -------
        The next bounds of the list
        """

        return super(BoundsList, self).__next__()

    def print(self):
        """
        Print the BoundsList to the console
        """

        raise NotImplementedError("Printing of BoundsList is not ready yet")


class QAndQDotBounds(Bounds):
    """
    Specialized Bounds that reads a model to automatically extract q and qdot bounds
    """

    def __init__(
        self,
        biorbd_model,
        q_mapping: BidirectionalMapping = None,
        q_dot_mapping: BidirectionalMapping = None,
    ):
        """
        Parameters
        ----------
        biorbd_model: biorbd.Model
            A reference to the model
        q_mapping: BidirectionalMapping
            The mapping of q
        q_dot_mapping: BidirectionalMapping
            The mapping of qdot. If q_dot_mapping is not provided, q_mapping is used
        """
        if biorbd_model.nbQuat() > 0:
            if q_mapping and not q_dot_mapping:
                raise RuntimeError(
                    "It is not possible to provide a q_mapping but not a q_dot_mapping if the model have quaternion"
                )
            elif q_dot_mapping and not q_mapping:
                raise RuntimeError(
                    "It is not possible to provide a q_dot_mapping but not a q_mapping if the model have quaternion"
                )

        if not q_mapping:
            q_mapping = BidirectionalMapping(range(biorbd_model.nbQ()), range(biorbd_model.nbQ()))

        if not q_dot_mapping:
            if biorbd_model.nbQuat() > 0:
                q_dot_mapping = BidirectionalMapping(range(biorbd_model.nbQdot()), range(biorbd_model.nbQdot()))
            else:
                q_dot_mapping = q_mapping

        q_ranges = []
        q_dot_ranges = []
        for i in range(biorbd_model.nbSegment()):
            segment = biorbd_model.segment(i)
            q_ranges += [q_range for q_range in segment.QRanges()]
            q_dot_ranges += [qdot_range for qdot_range in segment.QDotRanges()]

        x_min = [q_ranges[i].min() for i in q_mapping.to_first.map_idx] + [
            q_dot_ranges[i].min() for i in q_dot_mapping.to_first.map_idx
        ]
        x_max = [q_ranges[i].max() for i in q_mapping.to_first.map_idx] + [
            q_dot_ranges[i].max() for i in q_dot_mapping.to_first.map_idx
        ]

        super(QAndQDotBounds, self).__init__(min_bound=x_min, max_bound=x_max)


class InitialGuess(OptionGeneric):
    """
    A placeholder for the initial guess

    Attributes
    ----------
    init: PathCondition
        The initial guess

    Methods
    -------
    check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int)
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay
    concatenate(self, other: "InitialGuess")
        Vertical concatenate of two InitialGuess
    __bool__(self) -> bool
        Get if the initial guess is empty
    shape(self) -> int
        Get the size of the initial guess
    """

    def __init__(
        self,
        initial_guess: np.ndarray = (),
        interpolation: InterpolationType = InterpolationType.CONSTANT,
        **parameters,
    ):
        """
        Parameters
        ----------
        initial_guess: np.ndarray
            The initial guess
        interpolation: InterpolationType
            The type of interpolation of the initial guess
        parameters: dict
            Any extra parameters that is associated to the path condition
        """

        if isinstance(initial_guess, PathCondition):
            self.init = initial_guess
        else:
            self.init = PathCondition(initial_guess, interpolation=interpolation, **parameters)

        super(InitialGuess, self).__init__(**parameters)

    def check_and_adjust_dimensions(self, nb_elements: int, nb_shooting: int):
        """
        Sanity check if the dimension of the matrix are sounds when compare to the number
        of required elements and time. If the function exit, then everything is okay

        Parameters
        ----------
        nb_elements: int
            The expected number of rows
        nb_shooting: int
            The number of shooting points in the ocp
        """

        self.init.check_and_adjust_dimensions(nb_elements, nb_shooting, "InitialGuess")

    def concatenate(self, other: "InitialGuess"):
        """
        Vertical concatenate of two Bounds

        Parameters
        ----------
        other: InitialGuess
            The InitialGuess to concatenate with
        """

        self.init = PathCondition(
            np.concatenate((self.init, other.init)),
            interpolation=self.init.type,
        )

    def __bool__(self) -> bool:
        """
        Get if the InitialGuess is empty

        Returns
        -------
        If the InitialGuess is empty
        """

        return len(self.init) > 0

    @property
    def shape(self) -> int:
        """
        Get the size of the InitialGuess

        Returns
        -------
        The size of the InitialGuess
        """

        return self.init.shape


class InitialGuessList(UniquePerPhaseOptionList):
    """
    A list of InitialGuess if more than one is required

    Methods
    -------
    add(self, initial_guess: Union[PathCondition, np.ndarray], **extra_arguments)
        Add a new initial guess to the list
    __getitem__(self, i)
        Get the ith initial guess of the list
    __next__(self)
        Get the next initial guess of the list
    print(self)
        Print the InitialGuessList to the console
    """

    def add(self, initial_guess: Union[InitialGuess, np.ndarray], **extra_arguments):
        """
        Add a new initial guess to the list

        Parameters
        ----------
        initial_guess: Union[InitialGuess, np.ndarray]
            The initial guess to add
        extra_arguments: dict
            Any parameters to pass to the Bounds
        """

        if isinstance(initial_guess, InitialGuess):
            self.copy(initial_guess)
        else:
            super(InitialGuessList, self)._add(initial_guess=initial_guess, option_type=InitialGuess, **extra_arguments)

    def __getitem__(self, i):
        """
        Get the ith initial guess of the list

        Parameters
        ----------
        i: int
            The index of the initial guess to get

        Returns
        -------
        The ith initial guess of the list
        """

        return super(InitialGuessList, self).__getitem__(i)

    def __next__(self):
        """
        Get the next initial guess of the list

        Returns
        -------
        The next initial guess of the list
        """

        return super(InitialGuessList, self).__next__()

    def print(self):
        """
        Print the InitialGuessList to the console
        """
        raise NotImplementedError("Printing of InitialGuessList is not ready yet")
