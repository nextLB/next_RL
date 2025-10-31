"""
Logging and memory monitoring utilities
"""

import logging
import time
import gc
import psutil
import os
import torch
from contextlib import contextmanager
from typing import Tuple


def setupLogging(logLevel: int = logging.INFO, logFile: str = "./log/training.log"):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(logFile), exist_ok=True)

    logging.basicConfig(
        level=logLevel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logFile)
        ]
    )

    return logging.getLogger("Training")


class MemoryManager:
    """Memory manager for monitoring and optimizing memory usage"""

    @staticmethod
    def getGpuMemoryUsage() -> Tuple[float, float]:
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            cached = torch.cuda.memory_reserved() / 1024 ** 3
            return allocated, cached
        return 0.0, 0.0

    @staticmethod
    def getSystemMemoryUsage() -> float:
        """Get system memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3

    @staticmethod
    def clearMemory():
        """Clear memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memoryMonitor(operationName: str, logger: logging.Logger = None):
    """Context manager for memory monitoring"""
    startTime = time.time()
    gpuMemoryBefore = MemoryManager.getGpuMemoryUsage()
    systemMemoryBefore = MemoryManager.getSystemMemoryUsage()

    try:
        yield
    finally:
        endTime = time.time()
        gpuMemoryAfter = MemoryManager.getGpuMemoryUsage()
        systemMemoryAfter = MemoryManager.getSystemMemoryUsage()

        if logger:
            logger.debug(
                f"{operationName} - "
                f"Time: {endTime - startTime:.3f}s | "
                f"GPU Memory Change: {gpuMemoryAfter[0] - gpuMemoryBefore[0]:+.2f}GB | "
                f"System Memory Change: {systemMemoryAfter - systemMemoryBefore:+.2f}GB"
            )