"""
Post-training data processing and model retraining integration.

This module provides functions to hook into the game's training mode,
validating, appending, and retraining models with newly collected data.
"""
import logging
from typing import List, Dict, Any

# Lazy import to avoid circular dependencies
logger = logging.getLogger(__name__)


class PostTrainingProcessor:
    """
    Handles post-training data processing and model retraining.
    """

    def __init__(self):
        """
        Initialize the processor.
        """
        self._validator_and_trainer = None

    @property
    def validator_and_trainer(self):
        """Lazy load the validator and trainer to avoid circular imports"""
        if self._validator_and_trainer is None:
            # Import here to avoid circular dependencies
            from ai_platform_trainer.utils.data_validator_and_trainer import DataValidatorAndTrainer
            self._validator_and_trainer = DataValidatorAndTrainer()
        return self._validator_and_trainer
        
    def process_training_sequence(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Process training data from a completed training sequence.

        This method is designed to be called at the end of a training session
        to validate the newly collected data, append it to the existing dataset,
        and trigger retraining of the AI models.

        Args:
            training_data: List of data points collected during training

        Returns:
            Success status
        """
        if not training_data:
            logger.warning("No training data provided for processing")
            return False

        logger.info(f"Processing {len(training_data)} data points from training")
        
        # Use our validator and trainer to handle the data
        success = self.validator_and_trainer.process_new_data(training_data)
        
        if success:
            logger.info("Training data successfully processed and models retrained")
        else:
            logger.error("Failed to process training data")
            
        return success


# Singleton instance for easy access
post_training_processor = PostTrainingProcessor()
