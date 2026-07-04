# file: ai_platform_trainer/gameplay/input_handler.py
"""
InputHandler class for processing user input events.
"""
import logging

import pygame

from ai_platform_trainer.core.interfaces import IInputHandler


class InputHandler(IInputHandler):
    """
    Handles user input events, implementing the IInputHandler interface.
    """

    def __init__(self, event_callbacks=None):
        """
        Initialize the input handler.

        Args:
            event_callbacks: Dictionary mapping event types to callback functions
        """
        self.event_callbacks = event_callbacks or {}

    def register_callback(self, event_type, callback):
        """
        Register a callback function for a specific event type.

        Args:
            event_type: The pygame event type to register a callback for
            callback: The function to call when the event is received
        """
        self.event_callbacks[event_type] = callback

    def handle_input(self):
        """
        Process all pending input events.

        Returns:
            tuple: (bool, list) - First value indicates if game should continue,
                                second value is the list of events to be processed by states
        """
        events = list(pygame.event.get())
        should_continue = True

        for event in events:
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                should_continue = False

            # Call any registered callbacks for this event type
            if event.type in self.event_callbacks:
                result = self.event_callbacks[event.type](event)
                if result is False:  # Explicit check for False return value
                    should_continue = False

        return should_continue, events
