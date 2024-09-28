import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

"""
This file contains an implementation of DeepONets with a small example learning the Antiderivative operator.
Link to the dataset: https://yaleedu-my.sharepoint.com/personal/lu_lu_yale_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Flu%5Flu%5Fyale%5Fedu%2FDocuments%2Fdatasets%2Fdeepxde%2Fdeeponet%5Fantiderivative%5Faligned
Link to the paper: https://arxiv.org/abs/1910.03193
"""



class DataLoader:
    """
    X_train: A tuple of two PyTorch tensors. The first element has the shape (`N1`, `dim1`), and the second element has the shape (`N2`, `dim2`).
    y_train: A PyTorch tensor of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test, shuffle=True):
        if X_train[0].shape[0] != y_train.shape[0] or X_train[1].shape[0] != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the right format."
            )
        if X_test[0].shape[0] != y_test.shape[0] or X_test[1].shape[0] != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the right format."
            )
            
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.dataset_size = X_train[0].shape[0]
        self.shuffle = shuffle
        self.indices = np.arange(self.dataset_size)
        self.current_idx = 0
        
        if shuffle:
            np.random.shuffle(self.indices)

    def get_train_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        
        start_idx = self.current_idx
        end_idx = min(start_idx + batch_size, self.dataset_size)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        
        if self.current_idx >= self.dataset_size:
            self.current_idx = 0
            
        return (self.train_x[0][batch_indices], self.train_x[1]), self.train_y[batch_indices]

    def get_test_set(self):
        return self.test_x, self.test_y
    
    
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()
        self.activation = activation

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
    
    
class DeepONet(nn.Module):
    """
    layer_sizes_branch: A list of integers with layer widths for the branch net.
    layer_sizes_trunk (list): A list of integers with layer widths for the trunk net.
    The last layer width of branch and trunk net have to be the same.
    activation: (callable): An elementwise activation function.
    bias: (bool): whether to use a bias after merging or not.
    unstacked: (bool): whether to use a single branch net or one for each output dimension.
    """
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, bias=True, unstacked=True):
        super().__init__()
        self.activation = activation
        self.bias = bias
        self.unstacked = unstacked # if unstacked is false, stacked is true
        
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise ValueError("output dimension of branch net and trunk net must match!")
        
        self.p = layer_sizes_branch[-1]

        if unstacked:
            self.branch = FNN(layer_sizes_branch, self.activation)
        else:
            layer_sizes_branch = layer_sizes_branch[:-1] + [1]
            self.branches = []
            for _ in range(self.p):
                self.branches.append(FNN(layer_sizes_branch, self.activation))
        self.trunk = FNN(layer_sizes_trunk, self.activation)
        self.b = torch.nn.Parameter(torch.tensor(0.0)) if bias else None

    def merge_branch_trunk(self, x_func, x_loc):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        if self.bias:
            y += self.b
        return y

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        
        # forward the DeepONet
        if self.unstacked:
            branch_out = self.branch(x_func)
        else:
            branch_outs = []
            for i in range(self.p):
                branch_outs.append(self.branches[i](x_func))
            branch_out = torch.cat(branch_outs, dim=1)
            
        trunk_out = self.activation(self.trunk(x_loc))
        x = self.merge_branch_trunk(branch_out, trunk_out)
        
        return x


# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (torch.tensor(d["X"][0], dtype=torch.float), torch.tensor(d["X"][1], dtype=torch.float))
y_train = torch.tensor(d["y"], dtype=torch.float)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (torch.tensor(d["X"][0], dtype=torch.float), torch.tensor(d["X"][1], dtype=torch.float))
y_test = torch.tensor(d["y"], dtype=torch.float)

# Load into Dataloader
data = DataLoader(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
m = 100
dim_x = 1
net = DeepONet(
    [m, 40, 40],
    [dim_x, 40, 40],
    nn.functional.relu,
    bias=True,
    unstacked=True,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 1000
batch_size = 32

# Training loop
for epoch in range(1,epochs+1):
    net.train()  # Set the model to training mode

    for i in range(0, X_train[0].shape[0], batch_size):
        batch_x, batch_y = data.get_train_batch(batch_size)

        # Forward pass
        outputs = net(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights

    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}')

net.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_x, test_y = data.get_test_set() # ((N1, 100), (N2, 1)), (N1, N2)
    test_outputs = net(test_x) # (N1, N2)
    test_loss = criterion(test_outputs, test_y)
    print(f'Test Loss: {test_loss.item():.8f}')
    
    # plot example
    plt.plot(test_outputs[0,:], label="pred")
    plt.plot(test_y[0,:], label="ref")
    plt.legend()
    plt.show()
    
    


