# file: ai_platform_trainer/gameplay/state_machine.py
import logging
import pygame


class GameState:
    """Base class for all game states"""

    def __init__(self, game):
        """
        Initialize a new game state.

        Args:
            game: The main Game instance that this state belongs to
        """
        self.game = game

    def enter(self):
        """Called when entering this state"""

    def exit(self):
        """Called when exiting this state"""

    def update(self, delta_time):
        """
        Update logic for this state

        Args:
            delta_time: Time elapsed since last update in seconds

        Returns:
            str or None: Name of next state to transition to, or None to stay in current state
        """
        return None

    def render(self, renderer):
        """
        Render logic for this state

        Args:
            renderer: The renderer to use for drawing
        """

    def handle_event(self, event):
        """
        Handle pygame events for this state

        Args:
            event: The pygame event to handle

        Returns:
            str or None: Name of next state to transition to, or None to stay in current state
        """
        return None


class MenuState(GameState):
    """State for the main menu"""

    def enter(self):
        logging.info("Entering menu state")
        self.game.menu_active = True

    def exit(self):
        logging.info("Exiting menu state")
        self.game.menu_active = False

    def update(self, delta_time):
        # Menu doesn't need updates, as it's event-driven
        return None

    def render(self, renderer):
        self.game.menu.draw(self.game.screen)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            selected_action = self.game.menu.handle_menu_events(event)
            if selected_action:
                return self._handle_menu_selection(selected_action)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            selected_action = self.game.menu.handle_menu_events(event)
            if selected_action:
                return self._handle_menu_selection(selected_action)
        return None

    def _handle_menu_selection(self, selected_action):
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.game.running = False
            return None
        elif selected_action in ["play_learning"]:
            logging.info(f"'{selected_action}' selected from menu.")
            self.game.mode = selected_action
            return selected_action
        return None


class PlayLearningState(GameState):
    """State for gameplay in 'play_learning' mode"""

    def enter(self):
        logging.info("Entering play_learning state")
        # NOTE: This is a scaffold. We will replace this with the real logic.
        self.game.start_game("play_learning")

    def exit(self):
        logging.info("Exiting play_learning state")
        self.game.reset_game_state()

    def update(self, delta_time):
        current_time = pygame.time.get_ticks()
        if self.game.play_learning_mode_manager:
            self.game.play_learning_mode_manager.update(current_time)
        return None

    def render(self, renderer):
        learning_manager = self.game.play_learning_mode_manager
        renderer.render(self.game.menu, self.game.player, self.game.enemy, False, "play_learning", learning_manager)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                logging.info("Escape key pressed. Going to paused state.")
                return "paused"
            elif event.key == pygame.K_SPACE and self.game.player:
                self.game.player.shoot_missile(self.game.enemy.pos)
            elif event.key == pygame.K_m:
                logging.info("M key pressed. Returning to menu.")
                self.game.reset_g_state()
                return "menu"
            elif event.key == pygame.K_f:
                logging.debug("F pressed - toggling fullscreen.")
                self.game._toggle_fullscreen()
        return None


class PlayState(GameState):
    """State for gameplay in 'play' mode"""

    def enter(self):
        logging.info("Entering play state")
        # Initialize play mode
        self.game.player, self.game.enemy = self.game._init_play_mode()
        self.game.player.reset()
        self.game.spawn_entities()

        # Initialize play mode manager if needed
        if not hasattr(self.game, 'play_mode_manager') or self.game.play_mode_manager is None:
            from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
            self.game.play_mode_manager = PlayMode(self.game)

    def exit(self):
        logging.info("Exiting play state")

    def update(self, delta_time):
        current_time = pygame.time.get_ticks()

        # Handle respawn
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self.game.handle_respawn(current_time)

        # Use play mode manager for game updates
        self.game.play_mode_manager.update(current_time)

        return None

    def render(self, renderer):
        renderer.render(self.game.menu, self.game.player, self.game.enemy, False)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                logging.info("Escape key pressed. Going to paused state.")
                return "paused"
            elif event.key == pygame.K_SPACE and self.game.player:
                self.game.player.shoot_missile(self.game.enemy.pos)
            elif event.key == pygame.K_m:
                logging.info("M key pressed. Returning to menu.")
                self.game.reset_game_state()
                return "menu"
            elif event.key == pygame.K_f:
                logging.debug("F pressed - toggling fullscreen.")
                self.game._toggle_fullscreen()
        return None





class PausedState(GameState):
    """State for when the game is paused"""

    def enter(self):
        logging.info("Game paused")
        self.game.paused = True

    def exit(self):
        logging.info("Game unpaused")
        self.game.paused = False

    def update(self, delta_time):
        # No updates when paused
        return None

    def render(self, renderer):
        # First render the game underneath
        if self.game.mode == "play":
            self.game.states["play"].render(renderer)
        elif self.game.mode == "train":
            self.game.states["train"].render(renderer)

        # Then render pause overlay
        pause_font = pygame.font.SysFont(None, 64)
        pause_text = pause_font.render("PAUSED", True, (255, 255, 255))
        text_rect = pause_text.get_rect(
            center=(self.game.screen_width // 2, self.game.screen_height // 2)
        )
        self.game.screen.blit(pause_text, text_rect)

        # Render instructions
        instruction_font = pygame.font.SysFont(None, 24)
        instructions = [
            "ESC - Resume",
            "M - Main Menu"
        ]

        for i, instruction in enumerate(instructions):
            text = instruction_font.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(
                center=(self.game.screen_width // 2, self.game.screen_height // 2 + 50 + i * 30)
            )
            self.game.screen.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                logging.info("Escape key pressed. Resuming game.")
                return self.game.mode  # Return to the previous state (play or train)
            elif event.key == pygame.K_m:
                logging.info("M key pressed in pause menu. Returning to main menu.")
                self.game.reset_game_state()
                return "menu"
        return None


class GameOverState(GameState):
    """State for when the game is over"""

    def __init__(self, game):
        super().__init__(game)
        self.score = 0

    def enter(self):
        logging.info("Game over")
        # Calculate final score
        if hasattr(self.game.player, 'score'):
            self.score = self.game.player.score

    def exit(self):
        logging.info("Exiting game over state")

    def update(self, delta_time):
        return None

    def render(self, renderer):
        # Render game over screen
        self.game.screen.fill((0, 0, 0))

        # Game over text
        game_over_font = pygame.font.SysFont(None, 72)
        game_over_text = game_over_font.render("GAME OVER", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(
            center=(self.game.screen_width // 2, self.game.screen_height // 2 - 50)
        )
        self.game.screen.blit(game_over_text, text_rect)

        # Score
        score_font = pygame.font.SysFont(None, 48)
        score_text = score_font.render(f"Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(
            center=(self.game.screen_width // 2, self.game.screen_height // 2 + 20)
        )
        self.game.screen.blit(score_text, score_rect)

        # Instructions
        instruction_font = pygame.font.SysFont(None, 24)
        instructions = [
            "Press SPACE to play again",
            "Press M for main menu"
        ]

        for i, instruction in enumerate(instructions):
            text = instruction_font.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(
                center=(self.game.screen_width // 2, self.game.screen_height // 2 + 100 + i * 30)
            )
            self.game.screen.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Start a new game in the same mode
                return self.game.mode
            elif event.key == pygame.K_m:
                # Go back to the main menu
                return "menu"
        return None
