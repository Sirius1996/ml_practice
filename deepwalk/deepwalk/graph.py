import logging
import random
from time import time
from collections import defaultdict, Iterable
from six import iterkeys
from six.moves import range, zip, zip_longest


logger = logging.getLogger("deepwalk")

class Graph(defaultdict):

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()

        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):

        t0 = time()

        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0))

        self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    def random_walk(self, walk_length, alpha=0, rand=random.Random(), start=None):
        """
            Implement random walk.

            path_length: walk length.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < walk_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

# Generate random walk sequence
def build_deepwalk_corpus(G, num_paths, walk_length, alpha=0,
                          rand=random.Random(0)):
    walk_seq = []
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        # Generate random walk sequence for every node
        for node in nodes:
            walk_seq.append(G.random_walk(walk_length, rand=rand, alpha=alpha, start=node))

    return walk_seq

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist

# Load adjacencylist from file
def load_adjacencylist(file_, undirected=False, chunksize=10000):
    adjlist = []

    t0 = time()

    total = 0
    with open(file_) as f:
        for idx, adj_chunk in enumerate(map(parse_adjacencylist_unchecked, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
            total += len(adj_chunk)

    t1 = time()

    logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1 - t0))

    # Convert edge list into graph
    t0 = time()
    G = from_adjlist_unchecked(adjlist)
    t1 = time()

    logger.info('Converted edges to graph in {}s'.format(t1 - t0))

    if undirected:
        t0 = time()
        G = G.make_undirected()
        t1 = time()
        logger.info('Made graph undirected in {}s'.format(t1 - t0))

    return G