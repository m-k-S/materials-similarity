import torch
from torch_geometric.nn import global_mean_pool

def extract_features_egnn(network, n_layers, dataloader, device, save_path=False):
    embedding = []
    with torch.no_grad():
        for datum in dataloader:
            x = datum.pos.to(device)
            h = datum.x.to(device)
            y = torch.stack(list(datum.y.values())).to(device)
            edge_index = datum.edge_index.to(device)
            batch = datum.batch.to(device)
            
            # Forward pass
            e_batch = network.embedding(h)
            for i in range(0, n_layers):
                e_batch, x, _ = network._modules["e_block_%d" % i](e_batch, edge_index, x)
            e_batch = global_mean_pool(e_batch, batch)  # [batch_size, hidden_channels]

            embedding.append(e_batch)

    full_embed = torch.cat([e for e in embedding], dim=0)
    if save_path:
        torch.save(full_embed, save_path)
    return full_embed