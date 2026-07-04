# file: ai_platform_trainer/core/interfaces.py
"""
Abstract interfaces for key components in the AI Platform Trainer.
These interfaces define contracts that concrete implementations must follow.
"""
from abc import ABC, abstractmethod


class IInputHandler(ABC):
    """Interface for handling user input."""

    @abstractmethod
    def handle_input(self):
        """
        Handle user input events.

        Returns:
            tuple: (bool, list) - First value indicates if game should continue,
                               second value is the list of events to be processed by states
        """
