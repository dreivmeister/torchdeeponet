import numpy as np
import torch
from torch import nn

class BatchSampler:
    def __init__(self, dataset_size, shuffle=True):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.indices = np.arange(self.dataset_size)
        self.current_idx = 0
        
        if shuffle:
            np.random.shuffle(self.indices)
        
    def get_next(self, batch_size):
        # get next batch of indices of size batch_size
        if self.current_idx >= self.dataset_size:
            self.current_idx = 0
        
        start_idx = self.current_idx
        end_idx = min(start_idx + batch_size, self.dataset_size)
        batch_indices = self.indices[start_idx:end_idx]
        
        self.current_idx = end_idx
        return batch_indices
            

class TripleCartesianProd:
    """
    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if X_train[0].shape[0] != y_train.shape[0] or X_train[1].shape[0] != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if X_test[0].shape[0] != y_test.shape[0] or X_test[1].shape[0] != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(X_train[0].shape[0], shuffle=True)
        self.trunk_sampler = BatchSampler(X_train[1].shape[0], shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        
        indices = self.branch_sampler.get_next(batch_size)
        return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y
    
    
class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(
        self, layer_sizes, activation, regularization=None
    ):
        super().__init__()
        self.activation = activation
        self.regularizer = regularization

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
    
    
class DeepONetCartesianProd(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs = 1.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        num_outputs=1,
    ):
        super().__init__()
        self.activation = activation

        self.num_outputs = num_outputs

        self.branch = self.build_branch_net(layer_sizes_branch)
        self.trunk = self.build_trunk_net(layer_sizes_trunk)
        self.b = torch.nn.Parameter(torch.tensor(0.0))
        
        
    def build_branch_net(self, layer_sizes_branch):
        return FNN(layer_sizes_branch, self.activation)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation)

    def merge_branch_trunk(self, x_func, x_loc):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        
        # forward the DeepONet
        branch_out = self.branch(x_func)
        trunk_out = self.trunk(x_loc)
        x = self.merge_branch_trunk(branch_out, trunk_out)
        
        return x


# Load dataset
d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)

data = TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# Choose a network
m = 100
dim_x = 1
net = DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu"
)

# # Define a Model
# model = dde.Model(data, net)

# # Compile and Train
# model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
# losshistory, train_state = model.train(iterations=10000)

# # Plot the loss trajectory
# dde.utils.plot_loss_history(losshistory)
# plt.show()