"""Module to group similar hardware together in one unified class."""
from typing import Iterable
from typing import Sequence

import artiq.master.worker_db as manager


class VectorGrouper:
    """
    A class to group all passed objects together into one "meta-object".

    The "meta-object" functions as any of the passed objects, but groups all
    the outputs together so they function as one.

    Note: assumes all passed objects are descendants of the first.
    Tries to check if they do not, and throws an error if do not have the same type.

    Uses the first passed object as the model for what attributes are
    available on the other objects.

    The Facade software design pattern, but just passes through calls to inner objects.
    """

    def __init__(
        self, device_manager: manager.DeviceManager, wrapped_objs: Iterable[str]
    ):
        """
        Group passed devices together into one object that can all be called at once.

        Args:
            device_manager (:class:`DeviceManager`):
                a device manager that provides devices to group together.
                Can be modeled as a dictionary (must have .get() method).
            wrapped_objs (typing.Iterable[str]):
                names of the devices to group together (or a key to the devices).

        Raises:
            TypeError: If types of the passed objects are not compatible.
                Use same class for best results.
            KeyError: If a key in wrapped_objs parameter is not in device_manager

        """
        self.wrapped_obj_list = [
            device_manager.get(obj_name) for obj_name in wrapped_objs
        ]
        for i, obj in enumerate(self.wrapped_obj_list):
            if obj is None:
                raise KeyError(
                    "Device manager does not recognize {}".format(wrapped_objs[i])
                )
        for i, obj in enumerate(self.wrapped_obj_list[1:]):
            if not isinstance(obj, type(self.wrapped_obj_list[0])):
                raise TypeError(
                    "Passed different objects to {}."
                    "Type {} of '{}' does not match reference = {}".format(
                        self.__class__.__name__,
                        type(obj),
                        wrapped_objs[i],
                        type(self.wrapped_obj_list[0]),
                    )
                )

    @property
    def argument_length(self) -> Sequence[int]:
        """
        Return a list of the number of arguments supplied to each function call.

        When calling subfunctions, must provide either number of wrapped objects
        (i.e. this value)
        or 1 argument, which will be repeated as many times as needed.

        Returns:
            (Sequence[int]): the number of arguments accepted by function calls.

        """
        return (1, len(self.wrapped_obj_list))

    # todo: wrapping len() doesn't work
    # def __len__(self):
    #     return self.__getattr__("__len__")()

    def __getattr__(self, attr: str):
        """
        Wrap internal objects to group all their calls/properties together in a list.

        Args:
            attr (str): The attribute you would like to retrieve (function or property).
                Example: "abc".upper()

        Raises:
            ValueError: If the arguments that you provide are not the right shape.
                Arguments must either be Sequences of length self.argument_length
                (number of objects wrapped), or 1

        Returns:
            [function or property]: calls the desired attribute on each wrapped object.
                The function or property will be a list of the returns from
                the wrapped objects.

        """
        orig_attr = self.wrapped_obj_list[0].__getattribute__(attr)
        if callable(orig_attr):

            def vectorized(*args, **kwargs):
                args = list(args)  # convert from tuple to list

                # repeat arguments if not enough provided. Allow 1 or all args specified
                for i, arg in enumerate(args):
                    try:
                        arg_length = len(arg)
                    except TypeError:
                        arg_length = 1

                    if arg_length > 1 and arg_length != len(self.wrapped_obj_list):
                        raise ValueError(
                            "Invalid number of arguments provided. "
                            "Gave {}, expected {} (one per object)."
                            "Arg: {}".format(
                                arg_length, len(self.wrapped_obj_list), arg
                            )
                        )
                    elif arg_length == 1:
                        args[i] = len(self.wrapped_obj_list) * (arg,)  # repeat as tuple

                # setup args for each call. transposes arguments
                individual_call_args = tuple(map(list, zip(*args)))
                result = [
                    wo.__getattribute__(attr)(*individual_call_args[i], **kwargs)
                    for i, wo in enumerate(self.wrapped_obj_list)
                ]

                return result

            return vectorized
        else:
            return [wo.__getattribute__(attr) for wo in self.wrapped_obj_list]


# TODO: replace functionality w/ map() or itertools.starmap()
