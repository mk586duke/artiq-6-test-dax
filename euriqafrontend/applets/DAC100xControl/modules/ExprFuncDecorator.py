# *****************************************************************
# IonControl:  Copyright 2016 Sandia Corporation
# This Software is released under the GPL license detailed
# in the file "license.txt" in the top-level IonControl directory
# *****************************************************************
import logging
from collections import ChainMap

from .DataChanged import DataChangedS

SystemExprFuncs = dict()
UserExprFuncs = dict()
# ExpressionFunctions = dict()
ExpressionFunctions = ChainMap(UserExprFuncs, SystemExprFuncs)
NamedTraceDict = dict()
ExprFunUpdate = DataChangedS()
NamedTraceUpdate = DataChangedS()

_LOGGER = logging.getLogger(__name__)


def exprfunc(wrapped):
    fname = wrapped.__name__
    if fname in SystemExprFuncs:
        _LOGGER.error(
            "Function '%s' is already defined as an expression function!", fname
        )
    else:
        SystemExprFuncs[fname] = wrapped
    return wrapped


def userfunc(wrapped):
    fname = wrapped.__name__
    UserExprFuncs[fname] = wrapped
    return wrapped
