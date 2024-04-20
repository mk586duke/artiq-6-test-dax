import logging
import typing

import artiq.language.environment as artiq_env

_LOGGER = logging.getLogger(__name__)


class EURIQAModule(artiq_env.HasEnvironment):
    """EURIQA module class.

    Extends ARTIQ's :class:`HasEnvironment` by tracking the parent.
    This class is useful for finding modules attached to other children in the hierarchy
    and reusing them, instead of keeping track of conflicting state"""

    def __init__(self, managers_or_parent, *args, **kwargs) -> None:
        self.__managers_or_parent: artiq_env.HasEnvironment = managers_or_parent
        super().__init__(managers_or_parent, *args, **kwargs)

    def find_or_init_module(
        self, module_type: typing.Type, *args, auto_init: bool = True, **kwargs
    ) -> typing.List[typing.Any]:
        """Looks for a module in the class tree. If it does not exist, initializes it.

        If auto_init is not enabled, then will raise a RuntimeError.
        """

        found_modules = []
        for child in self.children:
            if isinstance(child, module_type):
                found_modules.append(child)

        # check super
        if not isinstance(self.__managers_or_parent, tuple):
            for child in self.__managers_or_parent.children:
                if isinstance(child, module_type):
                    found_modules.append(child)

        if len(found_modules) > 0:
            return found_modules
        elif auto_init:
            _LOGGER.debug(
                "Initializing Module %s with (args, kwargs) = (%s, %s)",
                module_type,
                args,
                kwargs,
            )
            return [module_type(*args, **kwargs)]
        else:
            raise RuntimeError(f"Could not find module of type {module_type}")

    def get_parent_attr_recursive(self, value: str) -> typing.List[typing.Any]:
        """Search for an attribute name recursively upwards.

        Essentially getattr(), but can search parents. Useful for when a value/function
        is defined at the top level, but needs to be found at a lower level.
        Effectively the inverse of super().
        """
        attrs = []
        p = self.__managers_or_parent

        class MissingValue(typing.NamedTuple):
            pass

        while isinstance(p, artiq_env.HasEnvironment):
            a = getattr(p, value, MissingValue)
            if a != MissingValue:
                _LOGGER.debug("Found attr %s in %s", value, p)
                attrs.append(a)
            try:
                p = p.__managers_or_parent
            except AttributeError:
                break

        return attrs
