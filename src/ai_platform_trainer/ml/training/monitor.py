"""
Training Monitor for Reinforcement Learning visualization.

This module provides real-time monitoring and visualization of RL training metrics
to help understand agent behavior and optimize training parameters.
"""
import os
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class TrainingMetrics:
    """Store and manage training metrics during RL model training."""

    def __init__(self):
        """Initialize the training metrics storage."""
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'training_steps': [],
            'avg_value_loss': [],
            'avg_policy_loss': [],
            'avg_entropy': [],
            'learning_rate': [],
            'exploration_rate': [],
            'timestamps': [],
            'fps': [],
        }
        self.behavioral_metrics = {
            'player_distances': [],
            'missile_avoidance': [],
            'movement_efficiency': [],
            'successful_hits': [],
        }
        self.start_time = time.time()

    def update(self, info_dict: Dict[str, Any]) -> None:
        """
        Update metrics with new training information.

        Args:
            info_dict: Dictionary containing metrics to update
        """
        current_time = time.time()
        self.metrics['timestamps'].append(current_time - self.start_time)

        # Update standard metrics
        for key, values in self.metrics.items():
            if key != 'timestamps' and key in info_dict:
                values.append(info_dict[key])

        # Update behavioral metrics
        for key, values in self.behavioral_metrics.items():
            if key in info_dict:
                values.append(info_dict[key])

    def get_metric(self, name: str) -> List[Any]:
        """
        Get a specific metric by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            List of values for the requested metric
        """
        if name in self.metrics:
            return self.metrics[name]
        elif name in self.behavioral_metrics:
            return self.behavioral_metrics[name]
        else:
            return []

    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """
        Get all metrics combined into a single dictionary.

        Returns:
            Dictionary containing all metrics
        """
        combined = {}
        combined.update(self.metrics)
        combined.update(self.behavioral_metrics)
        return combined

    def save(self, path: str) -> None:
        """
        Save metrics to a JSON file.

        Args:
            path: Path to save the metrics to
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert to a serializable format (convert numpy arrays to lists)
        serializable = {}
        for category in [self.metrics, self.behavioral_metrics]:
            for key, values in category.items():
                if values and isinstance(values[0], np.ndarray):
                    serializable[key] = [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in values
                    ]
                else:
                    serializable[key] = values

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logging.info(f"Saved training metrics to {path}")


class TrainingVisualizer:
    """Generate visualizations of training metrics."""

    def __init__(self, metrics: TrainingMetrics, output_dir: str):
        """
        Initialize the visualizer with metrics and output directory.

        Args:
            metrics: TrainingMetrics instance containing the data to visualize
            output_dir: Directory to save visualizations to
        """
        self.metrics = metrics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configure plots
        plt.style.use('ggplot')
        self.fig_size = (10, 6)

    def plot_rewards(self, window_size: int = 10) -> Figure:
        """
        Plot episode rewards over time with rolling average.

        Args:
            window_size: Size of the rolling average window

        Returns:
            Matplotlib figure object
        """
        rewards = self.metrics.get_metric('episode_rewards')
        if not rewards:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No reward data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return fig

        steps = self.metrics.get_metric('training_steps')

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(steps, rewards, 'b-', alpha=0.3, label='Episode Rewards')

        # Calculate and plot rolling average if enough data
        if len(rewards) >= window_size:
            rolling_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], rolling_avg, 'r-', label=f'{window_size}-Ep Avg')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Training Rewards Over Time')
        ax.legend()
        ax.grid(True)

        return fig

    def plot_losses(self) -> Figure:
        """
        Plot training losses over time.

        Returns:
            Matplotlib figure object
        """
        steps = self.metrics.get_metric('training_steps')
        value_loss = self.metrics.get_metric('avg_value_loss')
        policy_loss = self.metrics.get_metric('avg_policy_loss')
        entropy = self.metrics.get_metric('avg_entropy')

        if not steps or not value_loss:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No loss data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return fig

        fig, ax = plt.subplots(figsize=self.fig_size)

        if value_loss:
            ax.plot(steps, value_loss, 'b-', label='Value Loss')
        if policy_loss:
            ax.plot(steps, policy_loss, 'r-', label='Policy Loss')
        if entropy:
            ax.plot(steps, entropy, 'g-', label='Entropy')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss Value')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True)

        return fig

    def plot_learning_rate(self) -> Figure:
        """
        Plot learning rate over time.

        Returns:
            Matplotlib figure object
        """
        steps = self.metrics.get_metric('training_steps')
        learning_rate = self.metrics.get_metric('learning_rate')

        if not steps or not learning_rate:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No learning rate data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return fig

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(steps, learning_rate, 'g-')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True)

        return fig

    def plot_behavioral_metrics(self) -> Figure:
        """
        Plot behavioral metrics to analyze agent behavior.

        Returns:
            Matplotlib figure object
        """
        steps = self.metrics.get_metric('training_steps')
        distances = self.metrics.get_metric('player_distances')
        avoidance = self.metrics.get_metric('missile_avoidance')
        hits = self.metrics.get_metric('successful_hits')

        if not steps or (not distances and not avoidance and not hits):
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No behavioral data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return fig

        fig, ax = plt.subplots(figsize=self.fig_size)

        if distances:
            ax.plot(steps, distances, 'b-', label='Avg. Distance to Player')
        if avoidance:
            ax.plot(steps, avoidance, 'r-', label='Missile Avoidance Rate')
        if hits:
            ax.plot(steps, hits, 'g-', label='Player Hit Success Rate')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Metric Value')
        ax.set_title('Agent Behavioral Metrics')
        ax.legend()
        ax.grid(True)

        return fig

    def plot_training_speed(self) -> Figure:
        """
        Plot training speed (FPS) over time.

        Returns:
            Matplotlib figure object
        """
        timestamps = self.metrics.get_metric('timestamps')
        fps = self.metrics.get_metric('fps')

        if not timestamps or not fps:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No training speed data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return fig

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(timestamps, fps, 'b-')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Frames Per Second')
        ax.set_title('Training Speed')
        ax.grid(True)

        return fig

    def save_all_plots(self, timestamp: Optional[str] = None) -> None:
        """
        Generate and save all visualization plots.

        Args:
            timestamp: Optional timestamp to add to filenames
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plots = {
            "rewards": self.plot_rewards(),
            "losses": self.plot_losses(),
            "learning_rate": self.plot_learning_rate(),
            "behavior": self.plot_behavioral_metrics(),
            "speed": self.plot_training_speed()
        }

        for name, fig in plots.items():
            filename = f"{name}_{timestamp}.png"
            path = os.path.join(self.output_dir, filename)
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        logging.info(f"Saved all plots to {self.output_dir}")

    def generate_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Generate a combined dashboard visualization of training progress.

        Args:
            save_path: Optional path to save the dashboard image
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)

        # Add plots to the dashboard
        ax1 = fig.add_subplot(gs[0, 0])
        self._add_rewards_to_ax(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._add_losses_to_ax(ax2)

        ax3 = fig.add_subplot(gs[1, 0])
        self._add_behavior_to_ax(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self._add_learning_rate_to_ax(ax4)

        ax5 = fig.add_subplot(gs[2, :])
        self._add_speed_to_ax(ax5)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logging.info(f"Saved dashboard to {save_path}")

        return fig

    # Helper methods for dashboard generation
    def _add_rewards_to_ax(self, ax):
        """Add rewards plot to the given axis."""
        rewards = self.metrics.get_metric('episode_rewards')
        steps = self.metrics.get_metric('training_steps')

        if not rewards or not steps:
            ax.text(0.5, 0.5, 'No reward data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return

        ax.plot(steps, rewards, 'b-', alpha=0.3, label='Episode Rewards')

        # Calculate and plot rolling average
        window_size = min(10, len(rewards))
        if window_size > 1:
            rolling_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], rolling_avg, 'r-', label=f'{window_size}-Ep Avg')

        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)

    def _add_losses_to_ax(self, ax):
        """Add loss plots to the given axis."""
        steps = self.metrics.get_metric('training_steps')
        value_loss = self.metrics.get_metric('avg_value_loss')
        policy_loss = self.metrics.get_metric('avg_policy_loss')

        if not steps or (not value_loss and not policy_loss):
            ax.text(0.5, 0.5, 'No loss data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return

        if value_loss:
            ax.plot(steps, value_loss, 'b-', label='Value Loss', alpha=0.7)
        if policy_loss:
            ax.plot(steps, policy_loss, 'r-', label='Policy Loss', alpha=0.7)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)

    def _add_behavior_to_ax(self, ax):
        """Add behavioral metrics to the given axis."""
        steps = self.metrics.get_metric('training_steps')
        distances = self.metrics.get_metric('player_distances')
        hits = self.metrics.get_metric('successful_hits')

        if not steps or (not distances and not hits):
            ax.text(0.5, 0.5, 'No behavioral data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return

        if distances:
            # Normalize distances for better visualization
            max_dist = max(distances) if distances else 1
            norm_distances = [d/max_dist for d in distances]
            ax.plot(steps, norm_distances, 'b-', label='Norm. Distance', alpha=0.7)

        if hits:
            ax.plot(steps, hits, 'g-', label='Hit Rate', alpha=0.7)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Metric Value')
        ax.set_title('Agent Behavior')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)

    def _add_learning_rate_to_ax(self, ax):
        """Add learning rate plot to the given axis."""
        steps = self.metrics.get_metric('training_steps')
        learning_rate = self.metrics.get_metric('learning_rate')

        if not steps or not learning_rate:
            ax.text(0.5, 0.5, 'No learning rate data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return

        ax.plot(steps, learning_rate, 'g-')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate')
        ax.grid(True)

    def _add_speed_to_ax(self, ax):
        """Add training speed plot to the given axis."""
        timestamps = self.metrics.get_metric('timestamps')
        fps = self.metrics.get_metric('fps')

        if not timestamps or not fps:
            ax.text(0.5, 0.5, 'No training speed data available yet',
                    horizontalalignment='center', verticalalignment='center')
            return

        ax.plot(timestamps, fps, 'b-')
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('FPS')
        ax.set_title('Training Speed (higher is better)')
        ax.grid(True)


class TrainingMonitor:
    """
    Main class for monitoring and visualizing RL training.
    Combines metrics collection and visualization.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the training monitor.

        Args:
            output_dir: Directory to save visualizations and metrics to
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configure logging
        logger = logging.getLogger('training_monitor')
        logger.setLevel(logging.INFO)

        # Initialize metrics and visualizer
        self.metrics = TrainingMetrics()
        self.visualizer = TrainingVisualizer(self.metrics, output_dir)

        # Setup checkpointing
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 300  # 5 minutes

        logging.info(f"Training monitor initialized. Output directory: {output_dir}")

    def update(self, training_info: Dict[str, Any]) -> None:
        """
        Update training metrics with new information.

        Args:
            training_info: Dictionary containing training metrics
        """
        self.metrics.update(training_info)

        # Check if it's time to save checkpoint
        current_time = time.time()
        if current_time - self.last_checkpoint_time > self.checkpoint_interval:
            self.checkpoint()
            self.last_checkpoint_time = current_time

    def checkpoint(self) -> None:
        """Save metrics and generate visualization checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_path = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
        self.metrics.save(metrics_path)

        # Generate dashboard
        dashboard_path = os.path.join(self.output_dir, f"dashboard_{timestamp}.png")
        self.visualizer.generate_dashboard(dashboard_path)

        logging.info(f"Created training checkpoint at {timestamp}")

    def final_report(self) -> None:
        """Generate final report with all visualizations."""
        logging.info("Generating final training report...")

        # Save all individual plots
        self.visualizer.save_all_plots()

        # Save final metrics
        metrics_path = os.path.join(self.output_dir, "final_metrics.json")
        self.metrics.save(metrics_path)

        # Generate dashboard
        dashboard_path = os.path.join(self.output_dir, "final_dashboard.png")
        self.visualizer.generate_dashboard(dashboard_path)

        logging.info(f"Final training report saved to {self.output_dir}")
