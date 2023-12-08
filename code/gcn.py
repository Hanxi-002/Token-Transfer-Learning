import dgl
import pickle as pk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from dgl.nn import SAGEConv
import itertools
from sklearn.metrics import roc_auc_score
import random
import scipy.sparse as sp

# %% prepare for training
def split_edges(g, test_frac=0.1):
    u, v = g.edges()
    edge_ids = np.random.permutation(g.number_of_edges())
    

    # Split positive edges into training and testing sets
    num_test = int(len(edge_ids) * test_frac)
    train_size = len(edge_ids) - num_test
    test_pos_edges = edge_ids[:num_test]
    train_pos_edges = edge_ids[num_test:]

    # Create adjacency matrix
    adj_matrix = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    
    # Create negative edges
    non_edges = np.array(list(zip(*np.where(adj_matrix.toarray() == 0))))
    neg_eids = np.random.choice(len(non_edges), g.number_of_edges())
    
    # Split negative edges into training and testing sets
    test_neg_edges = neg_eids[:num_test]
    train_neg_edges = neg_eids[num_test:]

    # Extracting the respective u, v node indices for training and testing
    train_pos_u, train_pos_v = u[train_pos_edges], v[train_pos_edges]
    test_pos_u, test_pos_v = u[test_pos_edges], v[test_pos_edges]

    train_neg_u, train_neg_v = non_edges[train_neg_edges, 0], non_edges[train_neg_edges, 1]
    test_neg_u, test_neg_v = non_edges[test_neg_edges, 0], non_edges[test_neg_edges, 1]

    return (edge_ids, train_pos_u, train_pos_v, train_neg_u, train_neg_v,
            test_pos_u, test_pos_v, test_neg_u, test_neg_v)

# %%
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
# %%
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

# %% word & positional embedding
feats = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/tok_pos_mult_mat.pkl', 'rb'))

#load adj_mat for the lakh dataset
adj_mat = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/tok_pos_mult_cossim_mat.pkl', 'rb'))
#sns.histplot(adj_mat.flatten())
adj_mat = np.maximum(adj_mat, 0)
#sns.histplot(adj_mat.flatten())
#%% new data 
feats = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/embedding_token_mat.pkl', 'rb'))
#load adj_mat for the new dataset
adj_mat = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/new_token_adj_10000_window_samples.pkl', 'rb'))
#add an identity matrix to the adj_mat
adj_mat = adj_mat + np.identity(adj_mat.shape[0])
#%% baseline method
feats = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/embedding_token_mat.pkl', 'rb'))
# load adj_mat for the lak dataset, but just using token embeddings
adj_mat = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/token_embbeding_cossim.pkl', 'rb'))
adj_mat = np.maximum(adj_mat, 0)
#sns.histplot(adj_mat.flatten())
# %% make graph
#num_nodes=1564, num_edges=1269798
g = dgl.graph((torch.nonzero(torch.from_numpy(adj_mat), as_tuple=True)))
g.ndata['feats'] = torch.tensor(feats, dtype=torch.float32)
#%%
test_frac = 0.1
edge_ids, train_pos_u, train_pos_v, train_neg_u, train_neg_v, test_pos_u, test_pos_v, test_neg_u, test_neg_v = split_edges(g, test_frac)
train_g = dgl.remove_edges(g, edge_ids[:int(len(edge_ids) * test_frac)])
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

#%%
model = GraphSAGE(train_g.ndata['feats'].shape[1], 128)
pred = MLPPredictor(128)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

all_logits = []
all_loss = []
for e in range(500):
    # forward
    h = model(train_g, train_g.ndata['feats'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    all_loss.append(loss.item())
    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))

#save h in a pickle file
#pk.dump(h, open('/ix/djishnu/Hanxi/PGM_Project/GCN_base_embed.pkl', 'wb'))

#write loss to file
#pk.dump(all_loss, open('/ix/djishnu/Hanxi/PGM_Project/GCN_base_loss.pkl', 'wb'))
                       # %%
def find_similar_node(embedding, candidates, embeddings):
    """Find the node with the most similar embedding."""
    similarity = [np.dot(embedding, embeddings[candidate]) for candidate in candidates]
    return candidates[np.argmax(similarity)]

def get_neighbors(node, adjacency_matrix):
    """Get the neighbors of a node based on the adjacency matrix."""
    return np.where(adjacency_matrix[node] > 0)[0]

def generate_sequence(embeddings, adjacency_matrix, start_node, walk_length, restart_prob, top_n=5):
    """Generate a single biased random walk with a restart probability and randomness."""
    walk = [start_node]
    while len(walk) < walk_length:
        print(len(walk))
        if random.random() < restart_prob:
            walk = [start_node]
        else:
            cur = walk[-1]
            neighbors = get_neighbors(cur, adjacency_matrix)
            if len(neighbors) > 0:
                cur_embedding = embeddings[cur]
                # Find top N most similar nodes
                similarities = [(neighbor, np.dot(cur_embedding, embeddings[neighbor])) for neighbor in neighbors]
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_neighbors = [neighbor for neighbor, _ in similarities[:top_n]]
                # Randomly select from the top N neighbors
                next_node = random.choice(top_neighbors)
                walk.append(next_node)
            else:
                break  # No more neighbors to move to
    return walk


#cosine_mat = pk.load(open('/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/tok_pos_mult_cossim_mat.pkl', 'rb'))
#cosine_mat = np.maximum(adj_mat, 0)

num_nodes = adj_mat.shape[0]
walk_length = 30  # Length of the random walk
num_walks_per_node = 10  # Number of walks to start from each node
restart_prob = 0.2

starting_node = random.randint(0, num_nodes - 1)
all_walks = []
for _ in range(num_walks_per_node):
    walk = generate_sequence(h.detach().numpy(), adj_mat, starting_node, walk_length, restart_prob, top_n=5)
    all_walks.append(walk)

# write walks to file
# with open('/ix/djishnu/Hanxi/PGM_Project/word_walks.txt', 'w') as f:
#     for walk in all_walks:
#         f.write(','.join(map(str, walk)) + '\n')

# %%
