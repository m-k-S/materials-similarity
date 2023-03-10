from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np

# The average local property similarity of a datapoint 
# is the average of the property across its k nearest neighbors
# This method generates the ALPS score for each material in the dataset
def average_local_property_similarity_continuous(embedding, property, k=10):
    # data is an array or tensor of shape (n_datapoints, n_features)
    # property is an array or tensor of shape (n_datapoints, 1)
    # k is the number of nearest neighbors to consider

    # Get the k nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embedding)
    _, indices = neighbors.kneighbors(embedding) # indices is an array of shape (n_datapoints, k)

    alps = [(1/(k-1)) * (sum([property[indices[i, j]] for j in range(1, k)])) for i in range(len(embedding))]

    # Return the similarity scores
    return alps

def distinguishability(embedding, property, k=10):
    alps = average_local_property_similarity_continuous(embedding, property, k)
    return sum([1/(property[i] - alps[i]) for i in range(len(alps))])

# properties is a tensor of shape (n_datapoints, n_properties)
def transferability(embedding, properties, k=10):

    per_prop_standardized = []
    for prop in properties.transpose(0, 1): # (n_properties, n_datapoints)
        alps = average_local_property_similarity_continuous(embedding, prop, k) # (n_datapoints, 1)
        mu_prop = torch.mean(prop, dim=0)
        sigma_prop = torch.std(prop, dim=0)

        y_hat = abs((alps - mu_prop) / sigma_prop) # (n_datapoints, 1)
        per_prop_standardized.append(sum(y_hat))
    
    return sum(per_prop_standardized)

