import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

class LinearSchedule(LambdaLR):
    # Linear warmup and then linear decay learning rate scheduling
    # Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    # Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    def __init__(self, optimizer, t_total, warmup_steps=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(LinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

# Evaluate the model on the test set
def test(test_loader, network, criterion, device):
    network.eval()
    with torch.no_grad():
        total_mse = 0
        for datum in test_loader:
            x = datum.pos.to(device)
            h = datum.x.to(device)
            y = torch.stack(list(datum.y.values())).transpose(0, 1).to(torch.float32).to(device)
            edge_index = datum.edge_index.to(device)
            batch = datum.batch.to(device)

            outputs = network(h, x, edge_index, batch)

            mse = criterion(outputs[0], y)
            total_mse += mse.item()

        return total_mse / len(test_loader)

def train(train_loader, test_loader, network, num_epochs, init_lr, eval_interval, device):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=init_lr)
    scheduler = LinearSchedule(optimizer, num_epochs, warmup_steps=20)

    total_step = len(train_loader)

    train_losses = []
    test_error = []

    for epoch in range(1, num_epochs+1):
        for i, datum in enumerate(train_loader):  
            # Move tensors to the configured device
            x = datum.pos.to(device)
            h = datum.x.to(device)

            y = torch.stack(list(datum.y.values())).transpose(0,1).to(torch.float32).to(device)

            edge_index = datum.edge_index.to(device)
            batch = datum.batch.to(device)
            
            # Forward pass
            outputs = network(h, x, edge_index, batch)

            loss = criterion(outputs[0], y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            store_loss = loss.clone().detach()
            train_losses.append(store_loss)

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i+1, total_step, loss.item()), flush=True)

            if epoch % eval_interval == 0:
                with torch.no_grad():
                    # Get the accuracy of the model on the test set
                    test_mse = test(test_loader, network, criterion, device)
                    print ('Epoch [{}/{}], Test MSE: {:.4f}'.format(epoch, num_epochs, test_mse), flush=True)
                    test_error.append(test_mse)

        scheduler.step()

    return train_losses, test_error

def transfer(train_loader, test_loader, network, n_linear_layers, hidden_nf, out_node_nf, num_epochs, init_lr, eval_interval, device):
    # Reset layers
    for l in range(n_linear_layers):
        if l == n_linear_layers - 1:
            network._modules["lin_%d" % l] = nn.Linear(hidden_nf, out_node_nf)
        else: 
            network._modules["lin_%d" % l] = nn.Linear(hidden_nf, hidden_nf)

    for param in network.parameters():
        param.requires_grad = False
    for l in range(n_linear_layers):
        for param in network._modules["lin_%d" % l].parameters():
            param.requires_grad = True

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=init_lr)
    scheduler = LinearSchedule(optimizer, num_epochs, warmup_steps=20)

    total_step = len(train_loader)

    train_losses = []
    test_error = []

    for epoch in range(1, num_epochs+1):
        for i, datum in enumerate(train_loader):  
            # Move tensors to the configured device
            x = datum.pos.to(device)
            h = datum.x.to(device)

            y = torch.stack(list(datum.y.values())).to(torch.float32).transpose(0,1).to(device)

            edge_index = datum.edge_index.to(device)
            batch = datum.batch.to(device)
            
            # Forward pass
            outputs = network(h, x, edge_index, batch)

            loss = criterion(outputs[0], y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            store_loss = loss.clone().detach()
            train_losses.append(store_loss)

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i+1, total_step, loss.item()), flush=True)

            if epoch % eval_interval == 0:
                with torch.no_grad():
                    # Get the accuracy of the model on the test set
                    test_mse = test(test_loader, network, criterion, device)
                    print ('Epoch [{}/{}], Test MSE: {:.4f}'.format(epoch, num_epochs, test_mse), flush=True)
                    test_error.append(test_mse)

        scheduler.step()

    return train_losses, test_error