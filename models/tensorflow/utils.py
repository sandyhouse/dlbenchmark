# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.client import device_lib

def get_num_gpus():
    """Get the number of available GPUs"""
    local_device_protos = device_lib.list_local_devices()
    return sum([1 for x in local_device_protos if x.device_type == "GPU"])

def get_num_cpus():
    """Get the number of available CPUs"""
    local_device_protos = device_lib.list_local_devices()
    return sum([1 for x in local_device_protos if x.device_type == "CPU"])

def get_distribution_strategy(num_gpus, all_reduce_algorithm=None):
    """Get a Distribution Strategy used for training a model.
    Args:
        num_gpus: Number of GPUs to use.
        all_reduce_algorithm: Specify a all-reduce algorithm to use.
            Will choose based on device topology, if None.
    Returns:
        tf.contrib.distribute.DistributionStrategy object.
    """
    if num_gpus > get_num_gpus():
        raise ValueError("GPUs required greater than the number of available.")
    if num_gpus == 0:
        return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
    elif num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        if all_reduce_algorithm:
            return tf.contrib.distribute.MirroredStrategy(
                num_gpus=num_gpus,
                cross_tower_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
                    all_reduce_alg, num_packs=2))
        else:
            return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

def per_device_batch_size(batch_size, num_gpus):
    """Return batch_size per device."""
    if num_gpus <= 1:
        return batch_size
    if batch_size % num_gpus:
        raise ValueError("Batch size should be a multiple of the number "
                "of available GPUs.")
    return batch_size // num_gpus



if __name__ == "__main__":
    print(get_num_gpus())
    print(get_num_cpus())
    print(get_distribution_strategy(0))
    print(per_device_batch_size(256, 2))
