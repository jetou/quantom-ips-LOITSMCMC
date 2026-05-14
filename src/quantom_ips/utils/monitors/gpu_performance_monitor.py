import pynvml
import numpy as np
import torch

"""
Simple tool to extract the GPU utilization / memory consumption. The lines below were inspired by / partly taken from:
https://suzyahyah.github.io/code/pytorch/2024/01/25/GPUTrainingHacks.html
"""


# Access the resources and run inquiries:
def get_gpu_utilization(torch_device_id, disable_monitor):
    total_mem = -1.0
    used_mem = -1.0
    used_mem_fraction = -1.0
    utilization = -1.0

    if torch.cuda.is_available() and torch_device_id >= 0 and disable_monitor == False:
        # Init:
        pynvml.nvmlInit()
        handler = pynvml.nvmlDeviceGetHandleByIndex(torch_device_id)
        # Retreive information:
        info = pynvml.nvmlDeviceGetMemoryInfo(handler)
        # Memory:
        used_mem = info.used // 1024**2
        total_mem = info.total // 1024**2
        used_mem_fraction = np.round(used_mem * 100.0 / total_mem, 2)
        # Utilization:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handler).gpu
        # Shutdown:
        pynvml.nvmlShutdown()

    return {
        "gpu_total_mem": total_mem,
        "gpu_used_mem": used_mem,
        "gpu_used_mem_fraction": used_mem_fraction,
        "gpu_utilization": utilization,
    }


# Define function decorator / wrapper:
def mon_gpu_usage(torch_device_id, disable_monitor):
    def wrapper(func):
        def decorator(*arg, **kwargs):
            result = func(*arg, **kwargs)
            output = get_gpu_utilization(torch_device_id, disable_monitor)
            return result, output

        return decorator

    return wrapper
