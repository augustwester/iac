import numpy as np

class IACNet:
    def __init__(self, pools, min, _max, decay, rest):
        self.min, self.max, self.decay, self.rest = min, _max, decay, rest
        max_size = max([len(pool) for pool in pools])
        
        # aside from the user-specified pools, there must also be a pool of
        # "central" nodes that every excitatory connection should pass through.
        # the number of nodes in this pool should equal the size of the largest
        # user-specified pool.
        num_nodes = sum([len(pool) for pool in pools]) + max_size
        self.max_pool = [pool for pool in pools if len(pool) == max_size][0]
        self.nodes = list(set([unit for pool in pools for unit in pool]))
        
        self.W = np.zeros((num_nodes, num_nodes)) # weight matrix
        self.a = rest * np.ones((num_nodes, 1)) # activation vector
        
        # add within-pool inhibitory connections
        for pool in pools:
            for i, unit_i in enumerate(pool):
                for unit_j in pool[i+1:]:
                    self.add_connection(unit_i, unit_j, weight=-1, direct=True)
        
        # add "central" pool inhibitory connections
        inhib = -(np.ones((max_size, max_size)) - np.eye(max_size, max_size))
        self.W[-max_size:, -max_size:] = inhib
                    
    def add_connections(self, connections):
        for key, items in connections.items():
            for item in items:
                self.add_connection(key, item, weight=1)
                
    def add_connection(self, unit_i, unit_j, weight, direct=False):
        i, j = self.nodes.index(unit_i), self.nodes.index(unit_j)
        assert i != j, "Self-connections are not allowed"
        
        # all connections are bidirectional, meaning that connected nodes both
        # influence and are influenced by each other. this is what gives an IAC
        # network interesting dynamics.
        
        if direct:
            # within-pool inhibitory connections shouldn't pass through the
            # central nodes
            self.W[i,j], self.W[j,i] = weight, weight
        else:
            assert unit_i in self.max_pool, "Source must be in largest pool"
            i_max = self.max_pool.index(unit_i) + 1
            
            # connecting node from largest pool (e.g "apple") with node from
            # smaller pool (e.g. "fruit") through the intermediary central node
            # corresponding to (in this example) "apple".
            self.W[i, -i_max], self.W[-i_max, i] = weight, weight
            self.W[-i_max, j], self.W[j, -i_max] = weight, weight
    
    def input_and_cycle(self, inputs, num_cycles):
        k = np.array([self.nodes.index(k) for k in inputs])
        v = np.array([v for _, v in inputs.items()])
        self.a[k, 0] = v # set initial activations to specified input
        
        for _ in range(num_cycles):
            # the output of a node equals its activation when positive and 0
            # when negative (i.e. ReLU)
            net = self.W @ (self.a * (self.a > 0))
            
            # external input is added to net input in each cycle
            net[k, 0] += v
            
            # using a rate of 0.1 produces identical results to SimBrain's IAC
            # network, see https://www.youtube.com/watch?v=Nw3TEDfugLs
            self.a[net > 0] += 0.1*self.gt_update(self.a[net > 0], net[net > 0])
            self.a[net < 0] += 0.1*self.lt_update(self.a[net < 0], net[net < 0])
    
    def gt_update(self, a, net):
        return (self.max - a) * net - self.decay * (a - self.rest)
    
    def lt_update(self, a, net):
        return (a - self.min) * net - self.decay * (a - self.rest)

# using the example shown in https://www.youtube.com/watch?v=Nw3TEDfugLs
pools = [["apple", "pear", "zucchini", "broccoli", "snickers", "milky way"],
         ["healthy", "junk"],
         ["candy", "fruit", "vegetable"]]

connections = {"apple": ["healthy", "fruit"],
               "pear": ["healthy", "fruit"],
               "milky way": ["junk", "candy"],
               "zucchini": ["healthy", "vegetable"],
               "broccoli": ["healthy", "vegetable"],
               "snickers": ["junk", "candy"]}

net =  IACNet(pools, min=-1, _max=1, decay=0.05, rest=0)
net.add_connections(connections)
net.input_and_cycle({"apple": 1}, num_cycles=200)

for node in net.nodes:
    print(f"{node}: {net.a[net.nodes.index(node)]}")
for i, node in enumerate(net.max_pool):
    print(f"{node}[central]: {net.a[-(i+1)]}")