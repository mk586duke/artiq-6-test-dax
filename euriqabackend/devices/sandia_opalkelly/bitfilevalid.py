import os


class ConfigException(Exception):
    pass


def checkFileValid(filename, typeName, FPGAName):
    if not filename:
        raise ConfigException("No {0} specified".format(typeName))
    elif not isinstance(filename, str):
        raise ConfigException(
            "{0} '{1}' specified in '{2}' config is not a string".format(
                typeName, filename, FPGAName
            )
        )
    elif not os.path.exists(filename):
        raise ConfigException(
            "Unable to open {0} '{1}' specified in '{2}' config: Invalid {0} path".format(
                typeName, filename, FPGAName
            )
        )
