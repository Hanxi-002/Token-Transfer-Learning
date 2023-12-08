#%% set up paths
import os
import sys
os.chdir('/ix/djishnu/Hanxi/PGM_Project/')
sys.path.insert(1, '/ix/djishnu/Aaron/2_misc/PGM_Project/Full-MIDI-Music-Transformer')
#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import node2vec

#%%

# place holder for the GCN output
weighted_adjacency_matrix = np.array([
    [0, 0.5, 0, 0, 0.3],
    [0.5, 0, 0.7, 0, 0],
    [0, 0.7, 0, 0.9, 0],
    [0, 0, 0.9, 0, 0.4],
    [0.3, 0, 0, 0.4, 0]
])

# Convert the weighted adjacency matrix to a NetworkX graph
graph = nx.from_numpy_matrix(weighted_adjacency_matrix)


# Step 2: Apply Node2Vec
# Initialize Node2Vec model
# Note: You might want to experiment with the 'weight_key' parameter if your weights represent something other than the strength of connection
node2vec = node2vec.Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, weight_key='weight')

# Fit the model (this might take some time)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Generate node embeddings
embeddings = {node: model.wv[node] for node in graph.nodes()}

# Example: Get the embedding for a specific node
node_id = 2  # change to a valid node ID in your graph
node_embedding = embeddings[node_id]

print(f"Embedding for node {node_id}: {node_embedding}")

# %%
import random


def find_similar_node(embedding, candidates_embeddings):
    """Find the node with the most similar embedding."""
    similarity = [np.dot(embedding, candidate) for candidate in candidates_embeddings]
    return np.argmax(similarity)

def generate_biased_walk(model, graph, start_node, walk_length):
    """Generate a single biased random walk."""
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            cur_embedding = model.wv[str(cur)]
            neighbors_embeddings = [model.wv[str(neighbor)] for neighbor in neighbors]
            next_node = neighbors[find_similar_node(cur_embedding, neighbors_embeddings)]
            walk.append(next_node)
        else:
            break
    return walk

# Assuming 'model' is your trained Node2Vec model and 'graph' is the original graph
walk_length = 10  # Length of the random walk
num_walks_per_node = 5  # Number of walks to start from each node

# Generate walks
all_walks = []
for node in graph.nodes():
    for _ in range(num_walks_per_node):
        walk = generate_biased_walk(model, graph, node, walk_length)
        all_walks.append(walk)

# Now 'all_walks' contains your sequences

# %%
