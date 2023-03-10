from .graphs import get_features_and_coords
from matminer.datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# matminer datasets: https://hackingmaterials.lbl.gov/matminer/dataset_summary.html

# BoltzTraP documentation: https://hackingmaterials.lbl.gov/matminer/dataset_summary.html#boltztrap-mp
# MatBench documentation:  https://hackingmaterials.lbl.gov/automatminer/datasets.html#
 
def get_boltztrap_data():
    df = load_dataset("boltztrap_mp")
    df = df.drop(columns=['mpid', 'formula'])
    return df

def get_matbench_steels_data():
    df = load_dataset("matbench_steels")
    return df

def make_graphs(df, batch_size):
    dataset, largest_element = get_features_and_coords(df)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader, largest_element
