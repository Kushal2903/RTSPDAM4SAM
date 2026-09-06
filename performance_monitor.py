import pynvml
import torch
import threading
import time

class GPUPerformanceMonitor:
    def __init__(self):
        self.gpu_usage = 0
        self.memory_usage = 0
        self.monitoring = True
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except:
            self.available = False
            print("NVML not available - GPU monitoring disabled")
    
    def update(self):
        if not self.available:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.gpu_usage = util.gpu
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.memory_usage = (memory.used / memory.total) * 100
        except:
            pass
    
    def get_usage(self):
        self.update()
        return self.gpu_usage, self.memory_usage
