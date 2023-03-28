import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from tqdm import tqdm

periodic_table = pd.read_csv('data/periodic_table.csv')

# Find the largest element in the dataset
# Returns the atomic number of the largest element
def get_largest_element(df):
    largest_element = 0
    for idx, entry in df.iterrows():
        struct = entry.structure
        for site in struct._sites:
            symbol = str(list(site._species._data.keys())[0])
            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            if atomic_number > largest_element:
                largest_element = atomic_number
    return largest_element

# Create a fully connected graph from a pymatgen structure
# Returns an array of shape (2, n_edges) where each column is an edge
# This is the format required by the PyTorch Geometric library
def make_edge_indices(entry):
    n_nodes = len(entry.structure._sites)
    edge_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
    return torch.tensor(edge_index).transpose(0, 1)

# Create a graph from a Pandas DataFrame of pymatgen structures
# Returns a tuple of (feature_matrix, coord_matrix, label)
# feature_matrix is a matrix of shape (n_nodes, n_features)
# coord_matrix is a matrix of shape (n_nodes, 3)
# label is a scalar
def get_features_and_coords(df, largest_element=None):
    if largest_element is None:
        largest_element = get_largest_element(df)
    data = []

    for idx in tqdm(range(len(df)), desc="Building material graphs"):
        entry = df.iloc[idx]
        struct = entry.structure

        feature_matrix = []
        coord_matrix = []

        # Features
        for site in struct._sites:
            feature_vec = [0 for _ in range(largest_element)] # create a vector of zeros
            symbol = str(list(site._species._data.keys())[0])
            atomic_number = periodic_table.AtomicNumber[periodic_table['Symbol'] == symbol].values[-1]
            feature_vec[atomic_number - 1] = 1 # one-hot encode atomic number
            feature_matrix.append(feature_vec)

        # Coordinates
        for site in struct._sites:
            coords = site._frac_coords
            coord_matrix.append(coords)

        coord_matrix = torch.FloatTensor(np.array(coord_matrix))
        feature_matrix = torch.FloatTensor(np.array(feature_matrix))

        # Labels
        labels = {}
        for col in df.columns:
            if col != 'structure':
                labels[col] = torch.tensor(entry[col])

        if (feature_matrix is not None) and (len(feature_matrix) > 1): 
            edge_index=make_edge_indices(entry)
            if len(edge_index.shape) > 1:
                datum = Data(x=feature_matrix, edge_index=edge_index, y=labels, pos=coord_matrix)
                data.append(datum)

    return data, largest_element