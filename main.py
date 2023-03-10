import torch

from data import materials_project as mp
from models.egnn import EGNN
from train import train

# Hyperparameters
hidden_nf = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_labels = 6
init_lr = 0.01
eval_interval = 10
batch_size = 32
num_epochs = 100

df = mp.get_boltztrap_data()
train_loader, test_loader, largest_element = mp.make_graphs(df, batch_size)
n_feat = largest_element

egnn = EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf, out_node_nf=n_labels, in_edge_nf=0, device=device, normalize=True).to(device)

train(train_loader, test_loader, egnn, num_epochs, init_lr, batch_size, eval_interval, device)
