import numpy as np


class WeightUpdate:
    def __init__(self, post_id: int, delta_t: np.uint64):
        """A weight update event between two neurons
        Args:
            delta_t: time difference between pre and post synaptic spikes
        """
        self.delta_t = delta_t
        self.post_id = post_id
