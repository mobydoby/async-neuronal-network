class Spike:
    def __init__(self, origin_neuron, dest_id, time_sent, time_received, strength=1.0):
        """A spike event between two neurons
        Args:
            origin_neuron: the neuron that sent the spike
            time_sent: time the spike was sent
            time_received: time the spike was received
            strength: strength of the spike
        """
        self.origin_neuron = origin_neuron
        self.dest_id = dest_id
        self.time_sent = time_sent
        self.time_received = time_received
        self.strength = strength

    def __lt__(self, other):
        return self.time_received < other.time_received

    def __eq__(self, other):
        return self.time_received == other.time_received

    def __str__(self):
        if self.origin_neuron is None:
            return f"Input spike at time {self.time_received} with strength {self.strength}"
        return (
            f"N({self.origin_neuron.id}) sent spike at time {self.time_sent} "
            + f"--({self.strength})--> N({self.dest_id}) received at {self.time_received}"
        )

    def __repr__(self):
        if self.origin_neuron is None:
            return f"<Spike: Input --[{self.strength}]--> {self.dest_id} at {self.time_received}>"
        return f"<Spike: N({self.origin_neuron.id}) --[{self.strength}]--> N({self.dest_id}) at {self.time_received}>"
