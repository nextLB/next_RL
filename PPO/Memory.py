"""
内存管理相关工具
"""
import torch
import gc
import psutil
import os
import time
import logging
from contextlib import contextmanager
from typing import Tuple

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器，用于监控和优化内存使用"""

    @staticmethod
    def getGpuMemoryUsage() -> Tuple[float, float]:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            return allocated, cached
        return 0.0, 0.0

    @staticmethod
    def getSystemMemoryUsage() -> float:
        """获取系统内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3  # GB

    @staticmethod
    def clearMemory():
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memoryMonitor(operationName: str):
    """监控内存使用的上下文管理器"""
    startTime = time.time()
    gpuMemoryBefore = MemoryManager.getGpuMemoryUsage()
    systemMemoryBefore = MemoryManager.getSystemMemoryUsage()

    try:
        yield
    finally:
        endTime = time.time()
        gpuMemoryAfter = MemoryManager.getGpuMemoryUsage()
        systemMemoryAfter = MemoryManager.getSystemMemoryUsage()

        logger.debug(
            f"{operationName} - "
            f"耗时: {endTime - startTime:.3f}s | "
            f"GPU内存变化: {gpuMemoryAfter[0] - gpuMemoryBefore[0]:+.2f}GB | "
            f"系统内存变化: {systemMemoryAfter - systemMemoryBefore:+.2f}GB"
        )