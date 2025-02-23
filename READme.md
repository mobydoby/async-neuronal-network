# Async Neural Architecture Design

## Outline
    1. Purpose
    2. Background
    3. Requirements
    4. Design 
    5. Results
    6. Deviations
    7. Future Work
    8. References

## Purpose
At its core, this project explores how using temporal encoding can represent more information than traditional neural networks, with the intuition that more efficient representations can lead to smarter and more powerful inferences. This project aims to establish a foundational framework for creating asynchronous neural networks that can exploit temporal encoding for more efficient representations and computations.

## Background
A key difference in Artificial Deep Neural Networks (ANN) and Biological Neural Networks (Human and Animal Nervous Systems) is the lack of temporal encoding. Deep neural networks are restricted to process information one forward pass at a time. But biological intelligence networks are free activate asynchronously (for the most part, as we shall see). Research by Anthropic has shown using mechanistic interpretability that Deep Neural Networks act as sparse autoencoders for abstract concepts and meta information. This means that deep neural networks learn representations (compressed encodings) in neuron weight patterns naturally through back-propagation. This is a surprising and powerful discovery. It implies that a system with learnable representations are (nearly) all you need for powerful AI. All current implementations of Deep Neural Networks activate sequentially, layer by layer. This imposes a representation constraint - without workarounds, neuron activations can be used for maximally 1 representation per forward pass. 
From this key insight, this project attempts to build a foundational neuronal network that maximized representational efficiecy by 1) exploiting temporal encoding as seen in biological networks and 2) using vector embeddings encoding as seen in common deep neural networks. 
Exploring this paradigm of asynchronous neuron activations also sheds light on the nature of representation inside biolgical brains and may pioneer mechanistic interpretability for biological neural nets. Thus, the motivation of this project is biderectional; firstly from the frontiers of modeling biological neural networks, and at the same time, an attempt to equip artificial neural networks to exploit temporal encoding on top of existing vector embeddings for more efficient representations. 

The representational powers of artificial neural networks has been heavily explored in subfields of ML research like mechanistic interpretability. From a mathematical perspective, ANNs are universal function approximators with the power to fit any function using cascading sums of non-linear functions. These function approximators learn continuous vector representations of concepts and information. In addition, we see an emergent behavior of neural networks in accordance to scaling laws. Larger and deeper networks can represent exponentially more concepts. A prime example is examining the maximally activating image for neurons in particular neurons in a convolutional neural network. Neurons in early layers respond strongly to low level features such as edges and visual patterns. Neurons in deeper, subsequent layers activate maximally to more complex aggregated features such as as vague outlines of class label images (ex. dog, cat, car).

Prior work has also started exploring Spiking Neural Networks with success at a smaller scale. This project aims to create a type of "spiking neural network" with asynchronous modules to explore interpretability representation efficiency.

It's worth noting that with the exception of a few, spiking neural network implementations do not follow the maximal representation paradigm as described here. Notably, Numenta and the paper Spiking Neural Networks with Weighted Spikes have implemented a version of value and temporal encoding combined, but neither leverage asynchronous clusters of neurons. 

A naive approach which leads to the common mistake found in most implementations of spiking neural networks (SNNs) abandons quantitative vector embeddings and instead solely relies on discretized spike rate encoding. This severely limits the amount of imformation networks can represent by several orders of magnitude. 
To avoid this error, an analogous mechanism for granular value embedding must be maintained to supplement attempts to enhance representational power via temporal encoding.

Value encoding seems to have a natural analog that has been experimentally verified in multiple biolgical information networks. For instance, O'Keefe is know for his nobel prize winning work on the discovery of place cells. His subsequent work shows that the firing of place cells may be dual encoded together with the theta rhythm of EEG. 

Theta cycles are 8-12 Hz waves of electrical activity that are regulated by hyperpolarized cells in the thalamus. This can be thought of as a clocking mechanism to regulate areas of the brain, particularly the hippocampus for memory encoding and retrieval. 

Gamma cycles are high frequency electical cycles that occur in the brain. Furthermore, gamma cycles are nested within theta cycles with 5 - 8 gamma cycles occuring per theta period. In remarkable studies of place cells in mice, O'Keefe observed that place cells fired early and early with respect to theta phase as mice moved in the area. 

Theta and Gamma phase encoding is deeply reminiscent of positional encoding in the original transformer architecture.

Thus, this newly proposed framework maximizes the representational power through temporal encoding while crucially maintaining the previously overlooked representational advantages of artificial neural networks. 

The clocking that we see in the brain via repetitive inibitory signals sent from the thalamus is by definition, a synchronous mechanism. This observation contrasts our first intuition that asychronous activations is better than synchronous processing. There seems to be a fundamental advantage of processing sequential information since there is information that is baked into the temporal sequences of experience.

We seem to arrive at the conclusion that "Learnable representations is all you need" (for general intelligence). This gives me the motivation to implement a highly efficient learner of representations. 

## Requirements
### P0
A system that has mechanisms to able to encode information w.r.t. time.
### P0
A system that has mechanisms to continuosly represent information more efficiently (Learn)
### P0
A system that can initialize asynchronous groups of computational units. 
### P0
Must be able to efficiently initialize large groups of neurons with predefined connections
### P1
Must be a foundational service that creates building blocks for later experiments of asynchronous networks

## Design
### Overview
This Async Netork uses a real time simulator at its core. Since truly asynchronous event-driven neuron behavior is not possible without specialized hardware, a simulation using CPU/GPU was chosen. 
### Learning Method
A key difference in biological learning and ANNs is the learning method. ANNs use gradient descent with back propagation to update neuron to neuron connections/weights. 
This learning mechanism does not exist in biological neurons. Rather, according to neuroscience, neurons in the brain follow a learning rule called "Hebbian Learning", which can be summarized by the phrase: "Neurons that fire together, wire together". 
The proposed learning rule used periodic synapse updates. After N simulation cycles, all neurons will update their own synapses based on the Spike-timing dependent plasticity rule. This requires Neurons to also keep track of all upcoming synapse weight updates.
STDP was used to formulate the learning method of neurons. The learning rules for increased connectivity and decreased connectivity are described below. 
To keep this project in scope, 
### Where are the value embeddings?
As mentioned in the background section, the closest and most efficient implementation of value embeddings in spiking neural networks is to use a phase encoding relative to a local sine wave. This creates a constraint on this design. This means that neuron firings are without context unless there is a local encoding. We know that different areas of the brain use phase encoding differently so we leave the phase encoding as optional for each neuron to calculate spike output. 
### Neurons
Neurons are the core learning components in the network. They are responsible for maintaining their own incoming spikes, their own updates. While incoming spikes are passed from origin to destination. On the event of coincident firing, the neuron that received the coincident spike notifies the sending neuron to update its synaptic weight at the next update event. Particularly in large networks, synapses of neurons are connected locally in space. That is, each neuron only has synapses with other neurons that are close by in 3D space. This is an intentional bias in the design for 2 reasons: 
    1. The 


### Synapses
Synapses only exist in the context of their origin neurons. This reduces memory overhead by reducing the redundant information of origin/parent neurons. 
### Network
Neurons maintain their own synapse connections, updates, and update events in their respective data structures. The network serves as the interface between the environment and the internal computations of neuron graphs. In a large network, the 
Networks also are responsible for converting output spikes to interpretable actions for the action space. The choses implementation uses a weighted probabalistic interpretation of the last C simulation cycles 
The next question to answer is should and how inputs to the network clusters should be encoded. 
### Simulation
The simulator handles the clock cycle. During each clock cycle, Neurons handle all their spike and update weight events. All generated spikes are sent to be handled at the next simulation cycle. Actions are taken once during the end of each cycle. (This may be changed into an action buffer with the ability to take multiple actions per cycle)
### Environments

### Operational Metrics/Assumptions
1. Similation Clock Rate. Depending on the clock rate of the simulation, there is an inherent efficiency/accuracy tradeoff. For instance, if the simulation time steps are very small, it is uneccesary for each neuron to maintain the time of the last update to calculate exponential decay in each neuron. 
### Chosen Design
Python was chosen as a first pass for iteration speed and simplicity. 



## Results
The foundational modelling tools used to create neuronal networks with phase encoding has been completed. The core functionality including initializing groups of neurons, the learning mechanism (STDP), phase encoding using synchronous network clocks, and vectorized output

## Future Work
The following gives a "To-Do" list to build on top of this project. 
### Training and Testing on data. 
### GPU integration
Industry and Acadamia have not seen much success in spiking neural networks with asynchronous characteristics despite their theoretical benefifs because of the widespread success of current implementations of synchronous networks because of the continuing improvements of GPUs. 
### Tuning hyperparameters
### more timing granularity (precision)
### Visualization 

## Project Summary
This pet project was mostly a personal foray into computational neuroscience. There are some key insights I had while having conversations and thoughts about mechanistic interpretability and asynchronicity inside the human brain. The two main insights were that value encoding is exceptionally valuable and many implementations of spiking neural networks were taking the representational power of traditional neural networks for granted, AND that traditional neural networks are nowhere near efficient at representing/encoding information as efficiently as biological neural networks. This led to the hypothesis that biological intelligence is comprised of many asynchronous clusters of synchronous neurons within those clusters.

## References
    1. Toy Models of Representation
    2. Gamma Cycles
    3. Population Coding
    Furthermore, gamma cycles are nested within theta cycles with 5 - 8 gamma cycles occuring per theta period. 4. STDP