"""
A3C Trainer implementation
"""

import os
import time
from typing import List
import logging
from Config import A3CConfig, device
from A3CAgent import A3CAgent
from WorkerThread import A3CWorker


class A3CTrainer:
    """A3C Trainer"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

        # Training monitoring
        self.startTime = None

    def train(self):
        """Start training"""
        # Create model directory
        os.makedirs(self.config.modelSavePath, exist_ok=True)

        # Initialize global agent
        stateShape = (2, self.config.screenSize, self.config.screenSize)
        numActions = 4  # Breakout actions

        self.globalAgent = A3CAgent(stateShape, numActions, self.config)

        # Try to load existing model
        modelLoaded = self.globalAgent.loadModel(self.config.bestModelPath)

        # Create workers
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = A3CWorker(i, self.globalAgent, self.config)
            self.workers.append(worker)

        logging.info(f"Starting training with {self.config.numProcesses} worker threads")
        if modelLoaded:
            logging.info(f"Resuming from checkpoint, current global step: {self.globalAgent.globalStep}")

        # Start timer
        self.startTime = time.time()

        # Start workers
        for worker in self.workers:
            worker.start()
            time.sleep(0.5)  # Stagger start times

        # Monitor training progress
        try:
            lastGlobalStep = self.globalAgent.globalStep
            lastLogTime = time.time()

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                time.sleep(10)  # Check every 10 seconds

                currentStep = self.globalAgent.globalStep
                currentTime = time.time()

                # Calculate training speed
                if currentStep > lastGlobalStep:
                    stepsPerSec = (currentStep - lastGlobalStep) / (currentTime - lastLogTime)
                    elapsedTime = currentTime - self.startTime
                    remainingTime = (self.config.trainingTimesteps - currentStep) / stepsPerSec if stepsPerSec > 0 else 0

                    # Log progress
                    logging.info(f"Progress: {currentStep}/{self.config.trainingTimesteps} "
                               f"({currentStep/self.config.trainingTimesteps*100:.2f}%) | "
                               f"Speed: {stepsPerSec:.1f} steps/sec | "
                               f"Elapsed: {self.formatTime(elapsedTime)} | "
                               f"ETA: {self.formatTime(remainingTime)} | "
                               f"Episodes: {self.globalAgent.episodesCompleted} | "
                               f"Best Reward: {self.globalAgent.bestReward:.1f}")

                    lastGlobalStep = currentStep
                    lastLogTime = currentTime

                    # Save checkpoint
                    if currentStep % self.config.checkpointInterval == 0:
                        checkpointPath = os.path.join(
                            self.config.modelSavePath,
                            f"checkpoint_step_{currentStep}.pth"
                        )
                        self.globalAgent.saveModel(checkpointPath)
                        logging.info(f"Checkpoint saved at step {currentStep}")

                else:
                    logging.warning("No training progress detected...")

                    # Check thread status
                    aliveCount = sum(1 for w in self.workers if w.is_alive())
                    if aliveCount == 0:
                        logging.error("All worker threads have stopped!")
                        break

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training error: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            # Save final model
            finalModelPath = os.path.join(self.config.modelSavePath, "final_model.pth")
            self.globalAgent.saveModel(finalModelPath)

            # Wait for threads to finish
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            totalTime = time.time() - self.startTime
            logging.info(f"Training completed! Final global step: {self.globalAgent.globalStep}")
            logging.info(f"Best reward achieved: {self.globalAgent.bestReward:.1f}")
            logging.info(f"Total training time: {self.formatTime(totalTime)}")

    def formatTime(self, seconds: float) -> str:
        """Format time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"