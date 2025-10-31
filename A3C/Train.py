"""
Main training script
"""

import os
import torch
from Config import A3CConfig, device
from Log import setupLogging
from A3CTrainer import A3CTrainer


def A3C_PongNoFrameskip_v4_main():
    """Main training function"""
    try:
        # Setup logging
        logger = setupLogging()

        # Configuration
        config = A3CConfig()

        print("A3C Reinforcement Learning Training System")
        print("=" * 50)
        print(f"Environment: {config.environmentName}")
        print(f"Training Steps: {config.trainingTimesteps}")
        print(f"Number of Workers: {config.numProcesses}")
        print(f"Device: {device}")
        print("=" * 50)

        # Remove old model files to avoid compatibility issues
        if os.path.exists(config.bestModelPath):
            os.remove(config.bestModelPath)
            logger.info("Removed old model file to avoid compatibility issues")

        # Start training
        logger.info("Starting A3C training!")

        trainer = A3CTrainer(config)
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    A3C_PongNoFrameskip_v4_main()