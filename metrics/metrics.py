from sklearn.neighbors import NearestNeighbors
import numpy as np

# The average local property similarity of a datapoint is the average of the property across its k nearest neighbors
def average_local_property_similarity(data, labels, k=10):
    # Get the k nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit()

    # Get the average property of the neighbors
    avg_neighbor_property = np.mean([neighbor.y for neighbor in neighbors])

    # Get the average property of the datum
    avg_datum_property = datum.y

    # Return the similarity
    return 1 - np.abs(avg_neighbor_property - avg_datum_property)