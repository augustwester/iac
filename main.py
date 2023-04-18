import numpy as np

class IACNet:
    def __init__(self, pools, min, _max, decay, rest):
        self.min, self.max, self.decay, self.rest = min, _max, decay, rest
        max_size = max([len(pool) for pool in pools])
        num_nodes = sum([len(pool) for pool in pools]) + max_size
        self.max_pool = [pool for pool in pools if len(pool) == max_size][0]
        self.nodes = list(set([unit for pool in pools for unit in pool]))
        self.W = np.zeros((num_nodes, num_nodes))
        self.a = rest * np.ones((num_nodes, 1))
        
        # add within-pool inhibitory connections
        for pool in pools:
            for i, unit_i in enumerate(pool):
                for unit_j in pool[i+1:]:
                    self.add_connection(unit_i, unit_j, weight=-1, direct=True)
        
        # add within-instance inhibitory connections
        self.W[-max_size:, -max_size:] = -(np.ones((max_size, max_size))-np.eye(max_size, max_size))
                    
    def add_connections(self, connections):
        for key, items in connections.items():
            for item in items:
                self.add_connection(key, item, weight=1)
                
    def add_connection(self, unit_i, unit_j, weight, direct=False):
        i, j = self.nodes.index(unit_i), self.nodes.index(unit_j)
        assert i != j, "Self-connections are not allowed"
        
        if direct:
            self.W[i,j], self.W[j,i] = weight, weight
        else:
            assert unit_i in self.max_pool, "Source must be in largest pool"
            i_max = self.max_pool.index(unit_i) + 1
            self.W[i, -i_max], self.W[-i_max, i] = weight, weight
            self.W[-i_max, j], self.W[j, -i_max] = weight, weight
    
    def input_and_cycle(self, inputs, num_cycles):
        k = np.array([self.nodes.index(k) for k in inputs])
        v = np.array([v for _, v in inputs.items()])
        for i in range(num_cycles):
            net = self.W @ self.a
            net[k, 0] += v
            if i == 0:
                self.a[net > 0] += self.gt_update(self.a[net > 0], net[net > 0])
                self.a[net < 0] += self.lt_update(self.a[net < 0], net[net < 0])
            else:
                self.a[net > 0] += 0.1*self.gt_update(self.a[net > 0], net[net > 0])
                self.a[net < 0] += 0.1*self.lt_update(self.a[net < 0], net[net < 0])
            self.a = self.relu(self.a)
    
    def gt_update(self, a, net):
        return (self.max - a) * net - self.decay * (a - self.rest)
    
    def lt_update(self, a, net):
        return (a - self.min) * net - self.decay * (a - self.rest)
    
    def relu(self, x):
        x = x.copy()
        x[x < 0] = 0
        return x
        
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
net.input_and_cycle({"healthy": 1, "fruit": 1}, num_cycles=3884)

for node in net.nodes:
    print(f"{node}: {net.a[net.nodes.index(node)]}")