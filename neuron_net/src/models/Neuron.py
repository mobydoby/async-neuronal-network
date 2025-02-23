import numpy as np
import heapq
from typing import Iterator
from neuron_net.src.math.spiking_algorithms import (
    calc_spike_time,
    calc_weight_update,
    calc_next_potential,
)
from neuron_net.src.models.Spike import Spike
from neuron_net.src.models.WeightUpdate import WeightUpdate
import logging
import neuron_net.src.math.constants as constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = True


class Neuron:
    def __init__(
        self,
        id,
        is_input=False,
        is_output=False,
        rest=0,
        alpha=0.5,
        tau=25,
        threshold=0.15,
        gamma=200,
        synapses=None,
    ):
        """Initialize a Neuron
        Args:
            id: id of the neuron
            is_input: is this an input neuron
            is_output: is this an output neuron
            rest: resting potential of the neuron
            alpha: decay constant
            tau: time constant
            threshold: threshold for activation
            gamma: refractory period
            synapses: dictionary of connected neurons (post_neuron_id: weight)
        """
        self.id = id

        # Neuron parameters
        self.alpha = alpha
        self.tau = tau
        self.threshold = threshold
        self.gamma = gamma

        # Neuron info
        self._is_input = is_input
        self._is_output = is_output

        # Neuron state
        self._V_rest = rest
        self._V = rest
        # keeps track of how much decay happened since the last time
        self._time_of_last_update = 0.0
        # keeps track of the current spikes in this processing window (until next reset)
        self.curr_spikes = []
        self._num_spikes = 0
        # keeps track of time of last activation (used for refractory cooldown)
        self._time_of_last_activation = 0

        if synapses is None:
            synapses = {}

        self.synapses = (
            synapses  # dictionary of connected neurons (post_neuron_id: weight)
        )
        self.update_queue = []  # unordered queue for weight updates
        self.spike_queue = []  # priority queue for spike events

    def __str__(self):
        neuron_type = (
            "INPUT" if self._is_input else "OUTPUT" if self._is_output else "HIDDEN"
        )
        return (
            f"[{neuron_type} Neuron {self.id}] V={self._V:.1f}mV, {len(self.synapses)}"
            f" connections: {[(id, weight) for id, weight in self.synapses.items()]}"
        )

    def __repr__(self):
        neuron_type = (
            "INPUT" if self._is_input else "OUTPUT" if self._is_output else "HIDDEN"
        )
        return (
            f"[{neuron_type} Neuron {self.id}] V={self._V:.1f}mV, {len(self.synapses)}"
            f" connections: {[(id, weight) for id, weight in self.synapses.items()]}"
        )

    def get_num_spikes(self):
        return self._num_spikes

    def get_activation_encoding(self):
        """Return the spike times for this neuron in this current period"""
        return [spike.time_received for spike in self.curr_spikes]

    def get_time_of_last_activation(self):
        return self._time_of_last_activation

    def add_synapse(self, child_neuron_id: int, weight: float):
        """Connect this neuron to another neuron
        Args:
            child_neuron_id: id of the child neuron
            weight: weight of the connection
        """
        if child_neuron_id in self.synapses.keys():
            raise ValueError(
                f"[{self.id}] Neuron with id {child_neuron_id} already exists."
            )
        self.synapses[child_neuron_id] = weight

    def receive_spike(self, incoming_spike: Spike):
        """Receive a spike from a connected neuron
        Args:
            Spike: (origin, time_sent, time received, strength)
        """
        heapq.heappush(self.spike_queue, incoming_spike)

    def receive_weight_update(self, receiver_id: int, delta_t):
        """Queue a weight update event rom a post-synaptic neuron
        Args:
            receiver_id: id of the post-synaptic neuron
            delta_t: time difference between pre and post synaptic spikes
        """
        if receiver_id not in self.synapses.keys():
            raise ValueError(
                f"[{self.id}] Neuron with id {receiver_id} not found in synapses."
            )
        self.update_queue.append(WeightUpdate(receiver_id, delta_t))

    def process_spikes(
        self, time_cutoff: np.uint64, period_start_time, clock_period=100
    ) -> Iterator[Spike]:
        """Process all the spikes currently in the spike queue.
        Importantly, spikes do not represent information without the context of spike timing.
        Uses period_start_time and clock_period to encode the spike into continous values between -0.5 and 0.5.
        Args:
            time_cutoff: time to process spikes until
            period_start_time: start time of the current period. Used for time based encoding.
            clock_period: period of the clock cycle
        Yields:
            Spike: a spike event going to a post-synaptic neuron
        """
        logger.debug("inside process_spikes")
        heapq.heapify(self.spike_queue)
        # reset spike counter
        logger.debug("Resetting spike counter")
        self.curr_spikes = []
        self._num_spikes = 0
        while self.spike_queue:
            if self.spike_queue[0].time_received > time_cutoff:
                break
            spike = heapq.heappop(self.spike_queue)
            if spike.time_received < period_start_time:
                raise ValueError(
                    f"Received spike at {spike.time_received} before period start time {period_start_time}"
                )
            if spike.time_received - self._time_of_last_activation < self.gamma:
                logger.debug("in refractory...")
                # neuron is still in refractory
                delta = self._time_of_last_activation - spike.time_received
                if not self._is_input:
                    spike.origin_neuron.receive_weight_update(self.id, delta)
                continue

            # calculate amount of decay before spike
            self._V = calc_next_potential(
                spike.strength,
                self.tau,
                spike.time_received,
                self._time_of_last_update,
                self._V_rest,
                self._V,
            )

            if spike.origin_neuron is not None:
                logger.debug(
                    f"Potential after spike from N({spike.origin_neuron.id}) at {spike.time_received}: {self._V} mv"
                )
            else:
                logger.debug(f"Origin neuron is input")
            logger.debug(f"Potential after spike: {self._V}")

            # spike causes neuron potential to exceed threshold
            if self._V > self.threshold:
                logger.debug(
                    f"Spike at {spike.time_received} caused activation in Neuron {self.id}!"
                    + f" Entering refractory period..."
                )
                self.curr_spikes.append(spike)
                self._num_spikes += 1
                self._time_of_last_activation = spike.time_received
                # notify pre-synaptic neuron of spike
                if not self._is_input:
                    spike.origin_neuron.receive_weight_update(
                        self.id, spike.time_received - spike.time_sent
                    )
                # Spike next neurons, nothing happens if is_output neuron
                for neuron_id, weight in self.synapses.items():
                    # Calculate the time of the spike for all post-synaptic neurons
                    phase_ratio = spike.time_received - period_start_time / clock_period
                    yield Spike(
                        self,
                        neuron_id,
                        spike.time_received,
                        calc_spike_time(
                            weight,
                            spike.time_received,
                        ),
                        weight * phase_ratio,
                    )
                self._V = self._V_rest
            else:
                self._V += spike.strength

            self._time_of_last_update = spike.time_received

    def process_weight_updates(self) -> None:
        """Update all weights in the queue"""
        while self.update_queue:
            weight_update = self.update_queue.pop()
            new_weight = calc_weight_update(
                self.synapses[weight_update.post_id], weight_update.delta_t
            )
            self.synapses[weight_update.post_id] = new_weight
            if new_weight < 0:
                # pruning
                del self.synapses[weight_update.post_id]
