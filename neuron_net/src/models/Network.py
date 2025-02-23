from typing import Dict, List, Tuple
from neuron_net.src.models.Neuron import Neuron
from neuron_net.src.models.Spike import Spike
import numpy as np
from sklearn.preprocessing import normalize
from collections import deque
import warnings
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True


class Network:
    """Network is the main interface for the user to interact with the network.
    Networks use phase encoding to represent neuron spikes as continuous floating point values.
    The encoding function uses the curren
    |     |     |
    |     |     |
    |     |     |
    |     |     |
    """

    def __init__(
        self,
        neuron_connections: Dict[int, List[int]],
        input_list: List[int],
        output_list: List[int],
        # cycle_buffer_size=3,
        period_start_time=0,
        clock_cycle_period=100,  # ms - The rate that the encoder resets
        name="test-network",
    ):
        """Using a dictionary of neuron connections, initialize the network
        Args:
            neuron_connections: dictionary of neuron connections (neuron_id: [connected_neuron_ids])
            input_list: list of input neurons [neuron_ids]
            output_list: list of output neurons [neuron_ids]
            period_start_time: the time to start the phase encoding
            clock_cycle_period: the rate at which the encoder resets
            name: name of the network
        """
        self.neuron_connections = neuron_connections
        self.input_list = input_list
        self.output_list = output_list
        self.neurons = {}
        # initialize all neurons in this network.
        for neuron_id, connections in neuron_connections.items():
            if neuron_id in self.neurons:
                warnings.warn(f"Neuron with id {neuron_id} already exists")

            # create a neuron
            curr_neuron = Neuron(
                neuron_id,
                is_input=neuron_id in input_list,
                is_output=neuron_id in output_list,
            )

            # add synapses to the neuron
            for connection in connections:
                if connection not in self.neuron_connections:
                    raise ValueError(
                        f"Trying to connect {neuron_id} with {connection}, which is not found list of neurons"
                    )
                curr_neuron.add_synapse(connection, weight=0.2)
            self.neurons[neuron_id] = curr_neuron
        # ref start time allows the networks phase encoding to start/reset
        self.period_start_time = period_start_time
        self.clock_cycle_period = clock_cycle_period
        self.encoding_function = (
            lambda spike_time: (spike_time - period_start_time - 100)
            / clock_cycle_period
        )
        self.name = name

    def __str__(self):
        return (
            f"Network with {len(self.neurons)} neurons.\n"
            f" [{[n for n in self.neurons.values()]}]"
        )

    def __repr__(self):
        return (
            f"Network with {len(self.neurons)} neurons."
            f"\n  [{[n for n in self.neurons.values()]}]"
        )

    def clock(self, ref_start_time, clock_cycle_period=100):
        """clock, as in the verb, resets the network to the start time"""
        self.period_start_time = ref_start_time
        self.clock_cycle_period = clock_cycle_period

    def send_input_data(
        self, input_data: List[Tuple[int, np.float32]], curr_time
    ) -> None:
        """The input data is an array which indices correspond to neurons that get spiked
        Args:
            input_data: List of input data (Neuron_ID, Strength)
            curr_time: current time
        """
        for idx, strength in input_data:
            neuron = self.neurons[idx]
            inc_spike = Spike(
                origin_neuron=None,
                dest_id=idx,
                time_sent=None,
                time_received=curr_time,
                strength=strength,
            )
            neuron.receive_spike(inc_spike)

    def update(self, curr_time):
        """Update the network by processing all the spikes and weight updates"""
        for neuron in self.neurons.values():
            neuron.process_weight_updates()
            logging.debug(f"curr_time: {curr_time}")
            logging.debug(f"self.clock_cycle_period: {self.clock_cycle_period}")
            spikes = list(
                neuron.process_spikes(
                    curr_time, self.period_start_time, self.clock_cycle_period
                )
            )
            for spike in spikes:
                if spike.dest_id in self.neurons:
                    self.neurons[spike.dest_id].receive_spike(spike)
                else:
                    raise ValueError(f"Neuron {spike.dest_id} not found")
        # update the period reference time for proper phase encoding
        self.period_start_time += self.clock_cycle_period
        logging.warning(f"Updated period start time: {self.period_start_time}")

    def get_output(self) -> np.array:
        """Gets the activations of the output neurons"""
        # get the times of activation for all output neurons
        output_activation_times = [
            self.neurons[nid].get_activation_encoding() for nid in self.output_list
        ]
        output = np.zeros(shape=len(output_activation_times))
        for idx, output_neuron_spike_times in enumerate(output_activation_times):
            logging.debug(f"Output neuron spike times: {output_neuron_spike_times}")
            logging.debug(f"Output period start: {self.period_start_time}")
            logging.debug(f"clock: {self.clock_cycle_period}")
            output[idx] = sum(
                self.encoding_function(spike_time)
                for spike_time in output_neuron_spike_times
            )
        return output
