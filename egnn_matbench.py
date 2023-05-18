import torch
from matminer.datasets import load_dataset

from data import materials_project as mp
from models.egnn import EGNN
from train import train, transfer
from models.embeddings import extract_features_egnn
import metrics.metrics as metrics
from sklearn.model_selection import train_test_split

from torch_geometric.loader import DataLoader
from data.graphs import get_features_and_coords

import pickle

# Hyperparameters
hidden_nf = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_labels = 1
n_conv_layers = 3
n_linear_layers = 2
init_lr = 1e-3
eval_interval = 10
batch_size = 64
num_epochs = 250

### MATBENCH TRAINING ###

df = mp.get_matbench_mp_e_form_data()
train_loader, test_loader, largest_element = mp.make_graphs(df, batch_size, feature_size=94)
n_feat = largest_element

print ("Done processing graphs", flush=True)
print ("Number of materials: {}".format(len(train_loader)), flush=True)

egnn = EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf, out_node_nf=n_labels, device=device, n_conv_layers=n_conv_layers, n_linear_layers=n_linear_layers).to(device)

print ("Training EGNN on Matbench dataset", flush=True)
train_loss, test_loss = train(train_loader, test_loader, egnn, num_epochs, init_lr, eval_interval, device)

egnn_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader, device)

egnn_distinguishability = metrics.distinguishability(egnn_embedding, torch.tensor(df.e_form.values))

### JDFT2D TRANSFER ###

df_jdft2d = mp.get_matbench_jdft2d_data()
n_labels_jdft2d = 1

dataset_jdft2d, largest_element_jdft2d = get_features_and_coords(df_jdft2d)
train_set_jdft2d, test_set_jdft2d = train_test_split(dataset_jdft2d, test_size=0.2, random_state=42)
train_loader_jdft2d = DataLoader(train_set_jdft2d, batch_size=batch_size)
test_loader_jdft2d = DataLoader(test_set_jdft2d, batch_size=batch_size)

# Embedding without training
jdft2d_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader_jdft2d, device)

jdft2d_egnn_distinguishability_untrained = metrics.distinguishability(jdft2d_embedding, torch.tensor(df_jdft2d.exfoliation_en.values))

transfer_train_loss, transfer_test_loss = transfer(train_loader_jdft2d, test_loader_jdft2d, egnn, n_linear_layers, hidden_nf, n_labels_jdft2d, num_epochs, init_lr, eval_interval, device)

jdft2d_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader_jdft2d, device)

jdft2d_egnn_distinguishability_transfer = metrics.distinguishability(jdft2d_embedding, torch.tensor(df_jdft2d.exfoliation_en.values))

### JDFT2D TRAIN FROM SCRATCH ###
# Hyperparameters
hidden_nf = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_labels = 1
n_conv_layers = 3
n_linear_layers = 2
init_lr = 1e-3
eval_interval = 10
batch_size = 64
num_epochs = 250

train_loader_jdft2d = DataLoader(train_set_jdft2d, batch_size=batch_size)
test_loader_jdft2d = DataLoader(test_set_jdft2d, batch_size=batch_size)

egnn_2 = EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf, out_node_nf=n_labels, in_edge_nf=0, device=device, attention=True, normalize=True, n_conv_layers=n_conv_layers, n_linear_layers=n_linear_layers).to(device)

train_loss_jdft2d, test_loss_jdft2d = train(train_loader_jdft2d, test_loader_jdft2d, egnn_2, num_epochs, init_lr, eval_interval, device)

jdft2d_embedding_scratch = extract_features_egnn(egnn_2, n_conv_layers, train_loader_jdft2d, device)

jdft2d_egnn_distinguishability_scratch = metrics.distinguishability(jdft2d_embedding_scratch, torch.tensor(df_jdft2d.exfoliation_en.values))


# THINGS TO SAVE
save_data = {
    'egnn_train_loss': train_loss,
    'egnn_test_loss': test_loss,
    'egnn_embedding': egnn_embedding,
    'egnn_distinguishability': egnn_distinguishability,
    'jdft2d_egnn_distinguishability_untrained': jdft2d_egnn_distinguishability_untrained,
    'transfer_test_loss': transfer_test_loss,
    'transfer_train_loss': transfer_train_loss,
    'jdft2d_embedding': jdft2d_embedding,
    'jdft2d_egnn_distinguishability_transfer': jdft2d_egnn_distinguishability_transfer,
    'train_loss_jdft2d': train_loss_jdft2d,
    'test_loss_jdft2d': test_loss_jdft2d,
    'jdft2d_embedding_scratch': jdft2d_embedding_scratch,
    'jdft2d_egnn_distinguishability_scratch': jdft2d_egnn_distinguishability_scratch
}
    
with open('training_outputs.pkl', "wb") as f:
    pickle.dump(save_data, f)   
