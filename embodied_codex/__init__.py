"""RoboForge Embodied Coding Agent Harness.

The lightweight kernel is the default surface. The historical EvolutionEngine
is exposed lazily for the compatibility LIBERO runner and is not imported by
normal kernel users.
"""

from .interfaces import RobotDeployment
from .runtime import ControllerRuntime
from .workspace import TaskWorkspace
from .assets import (CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary,
                     SkillLibrary)

__version__ = "0.5.0"

def __getattr__(name):
    if name == "EvolutionEngine":
        from .evolution import EvolutionEngine
        return EvolutionEngine
    raise AttributeError(name)

__all__ = ["RobotDeployment", "ControllerRuntime", "TaskWorkspace",
           "CapabilityLibrary", "SkillLibrary", "ExperienceLibrary",
           "CapabilityGapLibrary", "EvolutionEngine"]
