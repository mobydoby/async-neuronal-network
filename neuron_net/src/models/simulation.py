from neuron_net.src.models.Network import Network
import numpy as np


class Simulator:
    def __init__(self, model, env, start_time, dt: np.uint64):
        self.model = model
        self.dt = dt
        self.time = start_time

    def step():
        pass


if __name__ == "__main__":
    # connections =
    input_list = [id for id in range(16)]
    output_list = [id for id in range()]
    net = Network()
