from neuron_net.src.models.Network import Network

"""
Clusters are networks that are synched together. They have access terminals to other neuron clusters.
"""


class Cluster(Network):
    def __init__(self, connections, input_list, output_list, name="test-cluster"):
        """
        Initialize a Cluster
        Args:
            connections: a dictionary of neuron connections
            input_list: list of input neurons
            output_list: list of output neurons
            clock_rate: the rate at which the determines phase codes
            name: name of the cluster
        """
        super().__init__(connections, input_list, output_list, clock_rate=10, name=name)
        self.clock_rate = clock_rate
