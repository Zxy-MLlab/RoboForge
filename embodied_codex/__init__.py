"""Legacy embodied runtime components retained for compatibility.

The canonical coding-agent entry point is :mod:`roboforge`; this package no
longer exports or advertises the historical generic loop.
"""

from .interfaces import RobotDeployment
from .kernel.runtime import ControllerRuntime
from .kernel.workspace import PersistentWorkspace
from .kernel.assets import (CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary,
                            SkillLibrary)
__version__ = "0.5.0"

__all__ = ["RobotDeployment", "ControllerRuntime", "PersistentWorkspace",
           "CapabilityLibrary", "SkillLibrary", "ExperienceLibrary",
           "CapabilityGapLibrary"]
