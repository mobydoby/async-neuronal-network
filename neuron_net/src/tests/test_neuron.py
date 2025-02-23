from neuron_net.src.models.Neuron import Neuron
from neuron_net.src.models.Spike import Spike
import pytest
import logging

"""Testing the core functionality of the Neuron Class:
- Adding synapses to a neuron
- Receiving spikes from other neurons
- Receiving weight updates from other neurons

- Processing spikes
    1. Receiving a Single Spike
    2. Recieve Multiple Spikes that Trigger Activation
- Processing weight updates
"""


@pytest.fixture
def neuron():
    return Neuron(0)


def test_add_synapse(neuron):
    neuron.add_synapse(1, weight=0.2)
    assert neuron.synapses[1] == 0.2


def test_add_existing_synapse(neuron):
    neuron.add_synapse(1, weight=0.2)
    with pytest.raises(ValueError):
        neuron.add_synapse(1, weight=0.3)


def test_receive_spike(neuron):
    n_1 = Neuron(1)
    neuron.receive_spike(Spike(n_1, 0, time_sent=1.0, time_received=1.1))
    assert len(neuron.spike_queue) == 1
    spike = neuron.spike_queue[0]
    assert spike.origin_neuron == n_1
    assert spike.time_received == 1.1
    assert spike.strength == 1.0


def test_receive_spike_after_time_cutoff(neuron, caplog):
    caplog.set_level(logging.DEBUG)
    n_1 = Neuron(1)
    neuron.receive_spike(Spike(n_1, 0, time_sent=1000, time_received=1210))
    neuron.receive_spike(Spike(n_1, 0, time_sent=1010, time_received=1211))
    spikes = list(neuron.process_spikes(time_cutoff=1200, period_start_time=1100))
    assert len(neuron.spike_queue) == 2
    assert neuron._V == neuron._V_rest
    assert neuron.get_time_of_last_activation() == 0  # default
    assert neuron.get_num_spikes() == 0


def test_process_spikes_spike(neuron, caplog):
    caplog.set_level(logging.DEBUG)
    n_1 = Neuron(1)
    n_1.add_synapse(0, weight=0.2)
    neuron.receive_spike(Spike(n_1, 0, time_sent=990, time_received=1050, strength=0.5))
    neuron.receive_spike(Spike(n_1, 0, time_sent=980, time_received=1015, strength=0.5))
    assert len(neuron.spike_queue) == 2
    spikes = list(neuron.process_spikes(time_cutoff=1100, period_start_time=1000))
    assert neuron.get_num_spikes() == 1
    assert neuron.get_time_of_last_activation() == 1050


def test_receive_weight_update_potentiated(neuron, caplog):
    n_1 = Neuron(1)
    n_1.add_synapse(0, weight=0.2)
    neuron.receive_spike(Spike(n_1, 0, time_sent=999, time_received=1011, strength=3.0))
    assert len(neuron.spike_queue) == 1
    spikes = list(neuron.process_spikes(time_cutoff=1100, period_start_time=1000))
    assert neuron.get_num_spikes() == 1
    # send back weight update
    assert len(n_1.update_queue) == 1
    n_1.process_weight_updates()
    assert n_1.synapses[0] > 0.2


def test_receive_weight_update_depressed(neuron, caplog):
    # start in refractory period
    n_1 = Neuron(1)
    n_1.add_synapse(0, weight=0.2)
    neuron.receive_spike(Spike(n_1, 0, time_sent=999, time_received=1011, strength=3.0))
    assert len(neuron.spike_queue) == 1
    spikes = list(neuron.process_spikes(time_cutoff=1100, period_start_time=1000))
    assert neuron.get_num_spikes() == 1
    # send back weight update
    assert len(n_1.update_queue) == 1
    n_1.process_weight_updates()
    assert n_1.synapses[0] > 0.2


def test_prune_synapses(neuron):
    """Initialize 2 pre-synaptic neurons, n_1 will be potentiated
    and n2 will be depressed. The depressed neuron should
    be pruned from the synapses of the post-synaptic neuron"""
    n_1 = Neuron(1)
    n_1.add_synapse(0, weight=0.2)
    n_2 = Neuron(2)
    n_2.add_synapse(0, weight=0.05)
    neuron.receive_spike(Spike(n_1, 0, time_sent=999, time_received=1011, strength=3.0))
    neuron.receive_spike(Spike(n_2, 0, time_sent=999, time_received=1012, strength=1.0))
    assert len(neuron.spike_queue) == 2
    next_spikes = list(neuron.process_spikes(time_cutoff=1100, period_start_time=1000))
    # after processing spikes
    assert len(n_1.update_queue) == 1
    assert len(n_2.update_queue) == 1
    n_1.process_weight_updates()
    n_2.process_weight_updates()
    assert n_1.synapses[0] > 0.2
    assert 0 not in n_2.synapses


def test_receive_weight_update_(neuron):
    with pytest.raises(ValueError):
        neuron.receive_weight_update(1, delta_t=0.1)


def test_process_spikes_no_spike(neuron, caplog):
    caplog.set_level(logging.DEBUG)
    n_1 = Neuron(1)
    neuron.receive_spike(
        Spike(n_1, 0, time_sent=1000, time_received=1100, strength=0.5)
    )
    neuron.time_of_last_update = 900
    spikes = list(neuron.process_spikes(time_cutoff=1200, period_start_time=1100))
    print(spikes)
    assert len(spikes) == 0
    assert neuron._num_spikes == 0
    assert neuron._V > neuron._V_rest


def test_process_weight_updates(neuron):
    neuron.add_synapse(1, weight=0.2)
    neuron.receive_weight_update(1, delta_t=0.1)
    neuron.process_weight_updates()
    assert len(neuron.update_queue) == 0
    assert neuron.synapses[1] != 0.2


def test_neuron_str_repr(neuron):
    assert str(neuron) == "[HIDDEN Neuron 0] V=0.0mV, 0 connections: []"
    assert repr(neuron) == "[HIDDEN Neuron 0] V=0.0mV, 0 connections: []"
