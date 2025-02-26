from neuron_net.src.models.Neuron import Neuron
from neuron_net.src.models.Network import Network
import numpy as np
import logging
import pytest


@pytest.fixture
def network_linear():
    """
    0 -> 1 -> 2 -> 3
    """
    net1_config = {
        "connections": {
            0: [1],
            1: [2],
            2: [3],
            3: [],
        },
        "input_list": [0],
        "output_list": [3],
    }

    period = 100
    return Network(
        net1_config["connections"],
        net1_config["input_list"],
        net1_config["output_list"],
        period_start_time=1000,
        clock_cycle_period=period,
        name="clocked_network_linear",
    )


@pytest.fixture
def network_tree():
    """
      -1
     / |
    0  3-(out)
    | /^
    2
    """
    net2_config = {
        "connections": {
            0: [1, 2],
            1: [3],
            2: [3],
            3: [],
        },
        "input_list": [0],
        "output_list": [3],
    }

    return Network(
        net2_config["connections"],
        net2_config["input_list"],
        net2_config["output_list"],
    )


@pytest.fixture
def network_loop():
    """
    0-> 1 -> 3
    | /^
    2
    """
    net2_config = {
        "connections": {
            0: [1, 2],
            1: [3],
            2: [1],
            3: [],
        },
        "input_list": [0],
        "output_list": [3],
    }

    return Network(
        net2_config["connections"],
        net2_config["input_list"],
        net2_config["output_list"],
    )


@pytest.fixture
def network_loop_clocked():
    """
    0-> 1 -> 3
    | /^
    2
    """
    net2_config = {
        "connections": {
            0: [1, 2],
            1: [3],
            2: [1],
            3: [],
        },
        "input_list": [0],
        "output_list": [3],
    }

    return Network(
        net2_config["connections"],
        net2_config["input_list"],
        net2_config["output_list"],
        period_start_time=1000,
        clock_cycle_period=100,
        name="clocked_network",
    )


def test_init_linear(network_linear):
    # asssert that the connections are correct
    neurons = network_linear.neurons
    assert len(neurons) == 4
    for n in range(4):
        assert n in neurons
        assert isinstance(neurons[n], Neuron)
    assert neurons[0].synapses == {1: 0.2}
    assert neurons[1].synapses == {2: 0.2}
    assert neurons[2].synapses == {3: 0.2}
    assert len(neurons[3].synapses) == 0


def test_init_tree(network_tree):
    # asssert that the connections are correct
    neurons = network_tree.neurons
    assert len(neurons) == 4
    for n in range(4):
        assert n in neurons
        assert isinstance(neurons[n], Neuron)
    assert neurons[0].synapses == {1: 0.2, 2: 0.2}
    assert neurons[1].synapses == {3: 0.2}
    assert neurons[2].synapses == {3: 0.2}
    assert len(neurons[3].synapses) == 0


def test_init_loop(network_loop):
    # asssert that the connections are correct
    neurons = network_loop.neurons
    assert len(neurons) == 4
    for n in range(4):
        assert n in neurons
        assert isinstance(neurons[n], Neuron)
    assert neurons[0].synapses == {1: 0.2, 2: 0.2}
    assert neurons[1].synapses == {3: 0.2}
    assert neurons[2].synapses == {1: 0.2}
    assert len(neurons[3].synapses) == 0


def test_send_input_data_linear(network_linear):
    network_linear.send_input_data([(0, 1.0)], 0)
    assert len(network_linear.neurons[0].spike_queue) == 1


def test_send_input_data_tree(network_tree):
    network_tree.send_input_data([(0, 1.0)], 0)
    assert len(network_tree.neurons[0].spike_queue) == 1


def test_send_input_data_loop(network_loop):
    network_loop.send_input_data([(0, 1.0)], 0)
    assert len(network_loop.neurons[0].spike_queue) == 1
    logging.debug(f"neuron 1 self._V: {network_loop.neurons[1]._V}")


def test_update_linear(network_linear):
    network_linear.send_input_data([(0, 0.5)], 1030)
    network_linear.send_input_data([(0, 2.0)], 1090)
    assert len(network_linear.neurons[0].spike_queue) == 2
    network_linear.update(1100)
    # no more spikes in queue
    assert len(network_linear.neurons[0].spike_queue) == 0
    # assert neuron spiked
    assert len(network_linear.neurons[0].curr_spikes) == 1
    # added to next neuron
    assert len(network_linear.neurons[1].spike_queue) == 1


def test_update_loop(network_loop, caplog):
    caplog.set_level(logging.DEBUG)
    network_loop.send_input_data([(0, 0.5)], 1030)
    network_loop.send_input_data([(0, 2.0)], 1090)
    assert len(network_loop.neurons[0].spike_queue) == 2
    network_loop.update(1100)
    # no more spikes in queue
    assert len(network_loop.neurons[0].spike_queue) == 0
    # assert neuron spiked
    assert len(network_loop.neurons[0].curr_spikes) == 1
    # added to next neuron
    assert len(network_loop.neurons[1].spike_queue) == 1


def test_phase_encoded_network(network_loop_clocked, caplog):
    caplog.set_level(logging.DEBUG)
    network_loop_clocked.send_input_data([(0, 0.5)], 1030)
    network_loop_clocked.send_input_data([(0, 2.0)], 1090)
    assert len(network_loop_clocked.neurons[0].spike_queue) == 2
    network_loop_clocked.update(1100)
    # no more spikes in queue
    assert len(network_loop_clocked.neurons[0].spike_queue) == 0
    # assert neuron spiked
    assert len(network_loop_clocked.neurons[0].curr_spikes) == 1
    # added to next neuron
    assert len(network_loop_clocked.neurons[1].spike_queue) == 1


def test_get_output(network_linear, caplog):
    caplog.set_level(logging.DEBUG)
    network_linear.clock(1000)
    network_linear.send_input_data([(0, 1.0)], 1080)
    network_linear.send_input_data([(0, 2.0)], 1090)
    logging.debug("Before 1100 Update")
    assert len(network_linear.neurons[0].spike_queue) == 2
    network_linear.update(1100)
    logging.debug("After 1100 Update")
    assert len(network_linear.neurons[0].spike_queue) == 0
    assert len(network_linear.neurons[1].curr_spikes) == 1
    logging.debug("Before 1200 Update")
    network_linear.update(1200)
    logging.debug("After 1200 Update")
    assert len(network_linear.neurons[2].curr_spikes) == 1
    rep = network_linear.get_output()
    logging.debug("HEOYYYY")
    logging.debug(rep)
