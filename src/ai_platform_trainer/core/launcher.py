"""
Standard Launcher Module (Adapter)

This is an adapter module that forwards to the canonical implementation
in engine/core/unified_launcher.py for backward compatibility.

DEPRECATED: Use ai_platform_trainer.engine.core.unified_launcher instead.
"""
import warnings
from ai_platform_trainer.engine.core.unified_launcher import main, launch_standard

# Add deprecation warning
warnings.warn(
    "Importing from ai_platform_trainer.core.launcher is deprecated. "
    "Use ai_platform_trainer.engine.core.unified_launcher instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backwards compatibility, provide the main function and forward to standard launcher
__all__ = ["main"]
