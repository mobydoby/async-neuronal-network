# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Phase-Encoding Asynchronous Neuronal Network** research project that implements spiking neural networks with temporal encoding. The core innovation combines:
- **Temporal encoding** (spike timing) with **value encoding** (vector embeddings)
- **Asynchronous neuron activations** with **phase-based encoding** using local clock cycles
- **Spike-Timing-Dependent Plasticity (STDP)** for learning instead of backpropagation

The project is inspired by biological neural networks (theta/gamma phase encoding in hippocampal place cells) and aims to maximize representational efficiency beyond traditional ANNs.

## Commands

### Testing
```bash
# Run all tests from project root
pytest

# Run all tests with coverage
pytest --cov=neuron_net --cov-report=html

# Run specific test file
pytest neuron_net/src/tests/test_network.py

# Run specific test
pytest neuron_net/src/tests/test_network.py::test_network_initialization

# Run tests with verbose logging (already configured in pyproject.toml)
pytest -s --log-cli-level=DEBUG
```

### Development
```bash
# Install in development mode (from project root)
pip install -e .

# Install with test dependencies
pip install -e ".[test]"
```

## Architecture

### Core Components

**Neuron** (`neuron_net/src/models/Neuron.py`)
- The fundamental computation unit maintaining its own state (voltage, spikes, synapses)
- Uses priority queue for spike events (heapq) and unordered queue for weight updates
- Implements leaky integrate-and-fire model with exponential decay
- Maintains refractory period and activation history
- Handles STDP weight updates autonomously

**Network** (`neuron_net/src/models/Network.py`)
- Main interface for creating and managing neuron graphs
- Initializes neurons from connection dictionaries
- Manages phase encoding using `period_start_time` and `clock_cycle_period`
- Routes spikes between neurons during `update()` cycles
- Converts output neuron spike times to continuous values via phase encoding function

**Spike** (`neuron_net/src/models/Spike.py`)
- Event objects representing spike transmission between neurons
- Contains: origin_neuron, dest_id, time_sent, time_received, strength
- Implements comparison operators for priority queue ordering

**Spiking Algorithms** (`neuron_net/src/math/spiking_algorithms.py`)
- `calc_spike_time()`: Computes spike arrival time based on weight and delay
- `calc_weight_update()`: STDP learning rule (potentiation if pre→post, depression if post→pre)
- `calc_next_potential()`: Leaky integrate-and-fire voltage update with exponential decay

### Key Design Patterns

**Time-based Simulation**: The network operates on discrete time steps. During each `update(curr_time)`:
1. All neurons process weight updates from previous cycle
2. Neurons process spikes from their queue up to curr_time
3. New spikes are generated and routed to destination neurons
4. `period_start_time` advances by `clock_cycle_period` for next phase encoding cycle

**Phase Encoding**: Spike arrival times within a clock period are encoded to continuous values:
```python
encoding_function = lambda spike_time: (spike_time - period_start_time - 100) / clock_cycle_period
```
This maps temporal information to vector representations (analogous to theta phase encoding in hippocampus).

**Graph-based Connectivity**: Networks are initialized with connection dictionaries:
```python
neuron_connections = {
    0: [1, 2],  # Neuron 0 connects to neurons 1 and 2
    1: [3],
    2: [3],
    3: []       # Output neuron
}
```

**STDP Learning**: Weight updates are queued when coincident spikes occur, then processed at periodic intervals. The learning rule strengthens connections when pre-synaptic spikes precede post-synaptic spikes (causality) and weakens them otherwise.

### Directory Structure
```
neuron_net/src/
├── models/          # Core neuron and network classes
│   ├── Neuron.py
│   ├── Network.py
│   ├── Spike.py
│   └── WeightUpdate.py
├── math/            # Mathematical functions for spiking and learning
│   ├── spiking_algorithms.py
│   └── constants.py
└── tests/           # Test suite
    ├── test_network.py
    ├── test_neuron.py
    └── conftest.py
```

## Development Notes

### Current State (as of March 2025)
- Core functionality is implemented: neuron initialization, STDP learning, phase encoding, spike routing
- NOT YET IMPLEMENTED: interfaces with training environments (Gym, Gazebo), visualization dashboard, GPU acceleration
- The project is in the foundational research phase

### Network Initialization Patterns
Input neurons receive external spikes via `send_input_data(input_data, curr_time)` where `input_data` is a list of `(neuron_id, strength)` tuples. Output is retrieved via `get_output()` which encodes output neuron spike times to continuous values.

### Testing Fixtures
Common network topologies are defined in `tests/conftest.py`:
- `network_linear`: Simple chain 0→1→2→3
- `network_tree`: Branching structure with multiple paths to output

### Logging
The codebase uses Python's logging module extensively. Tests are configured to show DEBUG-level logs (`--log-cli-level=DEBUG` in pytest options).
