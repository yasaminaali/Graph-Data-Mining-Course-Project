import networkx as nx
import numpy as np
import torch
import random


def build_graph(train_data, relationship_types):
    graphs = {rel: nx.DiGraph() for rel in relationship_types}

    for seq in train_data:
        for i in range(len(seq) - 1):
            for rel in relationship_types:
                if graphs[rel].get_edge_data(seq[i], seq[i + 1]) is None:
                    weight = 1
                else:
                    weight = graphs[rel].get_edge_data(seq[i], seq[i + 1])['weight'] + 1
                graphs[rel].add_edge(seq[i], seq[i + 1], weight=weight)

    for rel, graph in graphs.items():
        for node in graph.nodes:
            sum_weight = sum(graph.get_edge_data(j, i)['weight'] for j, i in graph.in_edges(node))
            if sum_weight != 0:
                for j, i in graph.in_edges(node):
                    graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum_weight)

    return graphs



def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)+1 
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)] 
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens] 
    return us_pois, us_msks, len_max 


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    # np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, attr, relationship_graphs, shuffle=False):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.relationship_graphs = relationship_graphs

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i, attr_data, taxo_data):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        alias_inputs, items, adjacency_matrices = [], [], []

        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + [0] * (len(np.unique(inputs)) - len(node)))
            
            A_list = []
            for graph in self.relationship_graphs.values():
                A = np.zeros((len(node), len(node)))
                for i in range(len(u_input) - 1):
                    if u_input[i + 1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i + 1])[0][0]
                    if graph.has_edge(u_input[i], u_input[i + 1]):
                        A[u][v] = graph.get_edge_data(u_input[i], u_input[i + 1])['weight']
                A_list.append(A)

            adjacency_matrices.append(A_list)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        return alias_inputs, adjacency_matrices, items, mask, targets
