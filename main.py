import torch
from matminer.datasets import load_dataset
from data.fingerprints import get_fingerprints
import numpy as np

from data import materials_project as mp
from models.egnn import EGNN
from train import train, transfer
from models.embeddings import extract_features_egnn
import metrics.metrics as metrics

from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from torch_geometric.loader import DataLoader
from data.graphs import get_features_and_coords

import pickle

# Hyperparameters
hidden_nf = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_labels = 6
n_conv_layers = 4
n_linear_layers = 2
init_lr = 0.01
eval_interval = 10
batch_size = 8
num_epochs = 100

df = mp.get_boltztrap_data()
train_loader, test_loader, largest_element = mp.make_graphs(df, batch_size, feature_size=94)
n_feat = largest_element

print ("Done processing graphs", flush=True)
print ("Number of materials: {}".format(len(train_loader)), flush=True)

egnn = EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf, out_node_nf=n_labels, in_edge_nf=0, device=device, normalize=True, n_conv_layers=n_conv_layers, n_linear_layers=n_linear_layers).to(device)

print ("Training EGNN on Boltztrap dataset", flush=True)
train_loss, test_loss = train(train_loader, test_loader, egnn, num_epochs, init_lr, eval_interval, device)

print ("Computing SOAP and MBTR fingerprints", flush=True)
soap, mbtr, labels = get_fingerprints(df)
print (soap.shape)
print (mbtr.shape)

print ("Extracting features from EGNN", flush=True)
egnn_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader, device)

print ("Computing metrics", flush=True)
egnn_distinguishability = metrics.distinguishability(egnn_embedding, torch.tensor(df.pf_n.values))
egnn_transferability = metrics.transferability(egnn_embedding, torch.tensor(df[[i for i in df.columns if i != 'structure']].values))

soap_distinguishability = metrics.distinguishability(soap, torch.tensor(df.pf_n.values))
soap_transferability = metrics.transferability(soap, torch.tensor(df[[i for i in df.columns if i != 'structure']].values))

mbtr_distinguishability = metrics.distinguishability(mbtr, torch.tensor(df.pf_n.values))
mbtr_transferability = metrics.transferability(mbtr, torch.tensor(df[[i for i in df.columns if i != 'structure']].values))

print ("EGNN Distinguishability: {}".format(egnn_distinguishability), flush=True)
print ("EGNN Transferability: {}".format(egnn_transferability), flush=True)

print ("SOAP Distinguishability: {}".format(soap_distinguishability), flush=True)
print ("SOAP Transferability: {}".format(soap_transferability), flush=True)

print ("MBTR Distinguishability: {}".format(mbtr_distinguishability), flush=True)
print ("MBTR Transferability: {}".format(mbtr_transferability), flush=True)

labels_as_array = np.stack([np.array(v) for v in labels.values()]).T
soap_train, soap_test, mbtr_train, mbtr_test, labels_train, labels_test = train_test_split(soap, mbtr, labels_as_array, test_size=0.2, random_state=42)

regressors = {
    'SOAP_XGB': XGBRegressor(tree_method="hist", n_estimators=64),
    # 'SOAP_KRR': KernelRidge(),
    'SOAP_MLP': MLPRegressor(),
    'MBTR_XGB': XGBRegressor(tree_method="hist", n_estimators=64),
    # 'MBTR_KRR': KernelRidge(),
    'MBTR_MLP': MLPRegressor()
}

mse = {}
for k,v in regressors.items():
    if k.startswith('SOAP'):
        v.fit(soap_train, labels_train)
        yhat = v.predict(soap_test)
        mse[k] = mean_squared_error(labels_test, yhat)
    elif k.startswith('MBTR'):
        v.fit(mbtr_train, labels_train)
        yhat = v.predict(mbtr_test)
        mse[k] = mean_squared_error(labels_test, yhat)

mse['EGNN'] = min(test_loss)
for k,v in mse.items():
    print ("{} MSE: {}".format(k, v))

df_jdft2d = load_dataset("matbench_jdft2d")
n_labels_jdft2d = 1

dataset_jdft2d, largest_element_jdft2d = get_features_and_coords(df_jdft2d)
train_set_jdft2d, test_set_jdft2d = train_test_split(dataset_jdft2d, test_size=0.2, random_state=42)
train_loader_jdft2d = DataLoader(train_set_jdft2d, batch_size=batch_size)
test_loader_jdft2d = DataLoader(test_set_jdft2d, batch_size=batch_size)

jdft2d_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader_jdft2d, device)
soap_jdft2d, mbtr_jdft2d, labels_jdft2d = get_fingerprints(df_jdft2d)

jdft2d_egnn_distinguishability_untrained = metrics.distinguishability(jdft2d_embedding, torch.tensor(df_jdft2d.exfoliation_en.values))
jdft2d_soap_distinguishability = metrics.distinguishability(soap_jdft2d, torch.tensor(df_jdft2d.exfoliation_en.values))
jdft2d_mbtr_distinguishability = metrics.distinguishability(mbtr_jdft2d, torch.tensor(df_jdft2d.exfoliation_en.values))

print ("JDFT2D EGNN Distinguishability (untrained): {}".format(jdft2d_egnn_distinguishability_untrained), flush=True)
print ("JDFT2D SOAP Distinguishability: {}".format(jdft2d_soap_distinguishability), flush=True)
print ("JDFT2D MBTR Distinguishability: {}".format(jdft2d_mbtr_distinguishability), flush=True)

transfer_train_loss, transfer_test_loss = transfer(train_loader_jdft2d, test_loader_jdft2d, egnn, n_linear_layers, hidden_nf, n_labels_jdft2d, num_epochs, init_lr, eval_interval, device)

labels_as_array = np.stack([np.array(v) for v in labels_jdft2d.values()]).T
soap_train_jdft2d, soap_test_jdft2d, mbtr_train_jdft2d, mbtr_test_jdft2d, labels_train_jdft2d, labels_test_jdft2d = train_test_split(soap_jdft2d, mbtr_jdft2d, labels_as_array, test_size=0.2, random_state=42)

regressors_jdft2d = {
    'SOAP_XGB_JDFT2D': XGBRegressor(tree_method="hist", n_estimators=64),
    # 'SOAP_KRR_JDFT2D': KernelRidge(),
    'SOAP_MLP_JDFT2D': MLPRegressor(),
    'MBTR_XGB_JDFT2D': XGBRegressor(tree_method="hist", n_estimators=64),
    # 'MBTR_KRR_JDFT2D': KernelRidge(),
    'MBTR_MLP_JDFT2D': MLPRegressor()
}

mse_jdft2d = {}
for k,v in regressors_jdft2d.items():
    if k.startswith('SOAP'):
        v.fit(soap_train_jdft2d, labels_train_jdft2d)
        yhat = v.predict(soap_test_jdft2d)
        mse[k] = mean_squared_error(labels_test_jdft2d, yhat)
    elif k.startswith('MBTR'):
        v.fit(mbtr_train_jdft2d, labels_train_jdft2d)
        yhat = v.predict(mbtr_test_jdft2d)
        mse[k] = mean_squared_error(labels_test_jdft2d, yhat)

mse_jdft2d['EGNN'] = min(transfer_test_loss)
for k,v in mse.items():
    print ("{} MSE: {}".format(k, v), flush=True)

jdft2d_embedding = extract_features_egnn(egnn, n_conv_layers, train_loader_jdft2d, device)

# Post-transfer learning

jdft2d_egnn_distinguishability_transfer = metrics.distinguishability(jdft2d_embedding, torch.tensor(df_jdft2d.exfoliation_en.values))
print ('JDFT2D EGNN Distinguishability (transfer): {}'.format(jdft2d_egnn_distinguishability_transfer), flush=True)

# DFT2D without transfer learning
# Trains from scratch

# Hyperparameters
hidden_nf = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_labels = 1
n_conv_layers = 4
n_linear_layers = 2
init_lr = 0.01
eval_interval = 10
batch_size = 8
num_epochs = 100

train_loader_jdft2d = DataLoader(train_set_jdft2d, batch_size=batch_size)
test_loader_jdft2d = DataLoader(test_set_jdft2d, batch_size=batch_size)

egnn_2 = EGNN(in_node_nf=n_feat, hidden_nf=hidden_nf, out_node_nf=n_labels, in_edge_nf=0, device=device, normalize=True, n_conv_layers=n_conv_layers, n_linear_layers=n_linear_layers).to(device)

train_loss_jdft2d, test_loss_jdft2d = train(train_loader_jdft2d, test_loader_jdft2d, egnn_2, num_epochs, init_lr, eval_interval, device)

jdft2d_embedding_scratch = extract_features_egnn(egnn_2, n_conv_layers, train_loader_jdft2d, device)

print (min(test_loss_jdft2d), flush=True)
jdft2d_egnn_distinguishability_scratch = metrics.distinguishability(jdft2d_embedding_scratch, torch.tensor(df_jdft2d.exfoliation_en.values))
print('JDFT2D EGNN Distinguishability (scratch): {}'.format(jdft2d_egnn_distinguishability_scratch), flush=True)

metrics_dict = {
    'EGNN Distinguishability': egnn_distinguishability,
    'EGNN Transferability': egnn_transferability,
    'SOAP Distinguishability': soap_distinguishability,
    'SOAP Transferability': soap_transferability,
    'MBTR Distinguishability': mbtr_distinguishability,
    'MBTR Transferability': mbtr_transferability,
    'JDFT2D EGNN Distinguishability (untrained)': jdft2d_egnn_distinguishability_untrained,
    'JDFT2D EGNN Distinguishability (transfer)': jdft2d_egnn_distinguishability_transfer,
    'JDFT2D EGNN Distinguishability (scratch)': jdft2d_egnn_distinguishability_scratch,
    'JDFT2D SOAP Distinguishability': jdft2d_soap_distinguishability,
    'JDFT2D MBTR Distinguishability': jdft2d_mbtr_distinguishability
}

# Saving everything:
save_data = {
    'EGNN_train_loss': train_loss,
    'EGNN_test_loss': test_loss,
    'SOAP embedding': soap,
    'MBTR embedding': mbtr,
    'EGNN embedding': egnn_embedding,
    'MSE': mse,
    'SOAP_JDFT2D': soap_jdft2d,
    'MBTR_JDFT2D': mbtr_jdft2d,
    'JDFT2D embedding': jdft2d_embedding,
    'JDFT2D transfer test loss': transfer_test_loss,
    'JDFT2D transfer train loss': transfer_train_loss,
    'JDFT2D MSE': mse_jdft2d,
    'JDFT2D train loss (scratch)': train_loss_jdft2d,
    'JDFT2D test loss (scratch)': test_loss_jdft2d,
    'JDFT2D embedding (scratch)': jdft2d_embedding_scratch,
    'metrics': metrics_dict
}

with open('training_outputs.pkl', "wb") as f:
    pickle.dump(save_data, f)