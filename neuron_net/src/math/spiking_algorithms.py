import numpy as np


def calc_spike_time(weight, time_delay, scale=100) -> np.uint64:
    """Calculate spike arrival time based on synaptic parameters.
    Uses a basic delay model where:
    arrival_time = pre_spike_time + base_delay / weight

    Parameters
    ----------
    pre_time : float
        Time of pre-synaptic spike
    weight : float
        Synaptic weight
    delay : float, optional
        Base delay value, by default 1.0

    Returns
    -------
    float
        Calculated arrival time of spike
    """
    return time_delay + weight * scale


def calc_weight_update(curr_weight: np.float64, delta_t: np.uint64) -> np.float64:
    """Update the weight of a synapse based on the time difference
    between pre and post synaptic spikes
    Args:
        curr_weight: current weight of the synapse
        delta_t: time difference between pre and post synaptic spikes
    """
    A_plus = 0.1  # Learning rate for potentiation
    A_minus = 0.12  # Learning rate for depression
    tau_plus = 20  # Time constant for potentiation
    tau_minus = 20  # Time constant for depression

    if delta_t < 0:
        # Pre-synaptic spike occurs after post-synaptic spike (depression)
        return curr_weight - A_minus * np.exp(delta_t / tau_minus)
    else:
        # Pre-synaptic spike occurs before post-synaptic spike (potentiation)
        return curr_weight + A_plus * np.exp(-delta_t / tau_plus)


def calc_next_potential(
    spike_strength, tau, time_received, time_of_last_update, V_rest, V_t, alpha=0.3
) -> np.float64:
    """Calculate the potential of the neuron after a spike"""
    dt = time_received - time_of_last_update
    decay = np.exp(-dt / tau)
    return V_t * np.exp(-dt / tau) + spike_strength * alpha
