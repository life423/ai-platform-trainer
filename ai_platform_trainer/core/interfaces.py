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


class IGameState(ABC):
    """Interface for game states in the state machine."""

    @abstractmethod
    def enter(self):
        """Called when entering this state."""

    @abstractmethod
    def exit(self):
        """Called when exiting this state."""

    @abstractmethod
    def update(self, delta_time):
        """
        Update logic for this state.

        Args:
            delta_time: Time elapsed since last update in seconds

        Returns:
            str or None: Name of next state to transition to, or None to stay in current state
        """

    @abstractmethod
    def render(self, renderer):
        """
        Render logic for this state.

        Args:
            renderer: The renderer to use for drawing
        """

    @abstractmethod
    def handle_event(self, event):
        """
        Handle events for this state.

        Args:
            event: The event to handle

        Returns:
            str or None: Name of next state to transition to, or None to stay in current state
        """


class IEntityFactory(ABC):
    """Interface for creating game entities."""

    @abstractmethod
    def create_player(self, screen_width, screen_height):
        """
        Create a player entity.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            A player entity
        """

    @abstractmethod
    def create_enemy(self, screen_width, screen_height):
        """
        Create an enemy entity.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            An enemy entity
        """
