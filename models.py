import torch
import pytorch_lightning as pl
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sigmoid, Tanh, ReLU, Module, Parameter, ParameterList, ModuleList, Sequential, Dropout,\
    L1Loss, BatchNorm1d
from torch.optim import AdamW
from torch_geometric.utils import degree
from torch_geometric.nn.glob import global_mean_pool
from .utils import L1Evaluator


class SingleLayerGNN(MessagePassing):
    def __init__(self, input_size):
        """
        A single layer that performs Ln * X, where:
        Ln = I - D^{-1/2} A D^{-1/2}
        """
        self.input_size = input_size
        super(SingleLayerGNN, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        # Do the normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg[deg == 0] = 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        return x - self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class PSN(Module):
    def __init__(self, layers, input_size, output_size, dropout_mlp=False, dropout_mlp_probability=0.1,
                 reduction='add'):
        """
        A block of many Single Layers without MLP layers in between.

        @param layers:                      Number of layers that this block will have.
        @param input_size:                  Size of the input nodes.
        @param output_size:                 Size of the output nodes.
        @param dropout_mlp:                 Whether to use dropout at the MLP in the output of the layer.
        @param dropout_mlp_probability:     The probability of dropout.
        """
        super(PSN, self).__init__()

        # Used during the model
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.relu = ReLU()

        # Type of reduction to use when updating
        self.reduction = reduction

        # Size of the input
        self.input_size = input_size

        # Dropout for the MLP at the output
        self.dropout_mlp = dropout_mlp
        self.dropout_mlp_probability = dropout_mlp_probability

        # Parameters used for the polynomial
        self.k_values = ParameterList([Parameter(torch.randn((1, 1))) for _ in range(layers)])

        self.num_layers = layers

        # Value for the constant for weighting each option (residual connection)
        self.weighting = Parameter(torch.tensor([0.0]))

        # Layers for the update
        self.layers = ModuleList([SingleLayerGNN(input_size) for _ in range(layers)])

        # Set the dropout
        if self.dropout_mlp:
            self.transformation = Sequential(Dropout(self.dropout_mlp_probability),
                                             Linear(input_size, output_size, bias=True),
                                             ReLU())
        else:
            self.transformation = Sequential(Linear(input_size, output_size, bias=True), ReLU())

    def forward(self, x, edge_index):
        # Save the original value for x
        x = x.float()
        original_x = x.clone()

        # Value that is going to update according to the polynomial
        update = 0

        # sum(gamma_k * L^k * X)
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            if self.reduction == 'add':
                update += (self.tanh(self.k_values[index]) * x.unsqueeze(1))
            elif self.reduction == 'mean':
                update += (self.tanh(self.k_values[index]) * x.unsqueeze(1)) / len(self.layers)

        # Average over the filters (always size 1 in this case)
        update = torch.mean(update, dim=1)

        # Constant that weights original value for x and updates
        constant = self.sigmoid(self.weighting)

        # Get the final output
        return self.transformation(constant * update + (1 - constant) * original_x)


class PSNLightning(pl.LightningModule):
    def __init__(self, filter_blocks, layers_per_filter, input_size, classes, lr, embedding_size=128,
                 dropout_mlp=False, dropout_mlp_probability=0.3, weight_decay=0.01):
        """
        Multilayer GNN with filters (or blocks) of Single layers, with linear transformations between each block.
        Performs graph classification.

        @param filter_blocks:      number of filters (or blocks)
        @param layers_per_filter:  number of single layers per filter block
        @param input_size:      size of input nodes
        @param classes:         number of classes (for graph classification)
        @param lr:              learning rate
        """
        super(PSNLightning, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_mlp = dropout_mlp
        self.dropout_mlp_probability = dropout_mlp_probability

        # GNN Layers
        self.input_gnn = PSN(layers_per_filter, input_size, embedding_size,
                             dropout_mlp=self.dropout_mlp,
                             dropout_mlp_probability=self.dropout_mlp_probability)
        self.layers = ModuleList([PSN(layers_per_filter, embedding_size, embedding_size,
                                      dropout_mlp=self.dropout_mlp,
                                      dropout_mlp_probability=self.dropout_mlp_probability) for _
                                  in range(filter_blocks - 1)])

        # Normalization layers
        self.input_norm = BatchNorm1d(input_size)
        self.normalization_layers = ModuleList([BatchNorm1d(embedding_size) for _ in range(filter_blocks - 1)])

        # Used in the model
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        # Task to solve
        self.task = 'regression'

        # Loss to use
        self.loss = L1Loss()

        # Metric to use
        self.evaluator = L1Evaluator()

        # Classification neural network
        if self.dropout_mlp:
            self.classification = Sequential(
                Dropout(self.dropout_mlp_probability),
                Linear(embedding_size, embedding_size),
                BatchNorm1d(embedding_size),
                ReLU(),
                Dropout(self.dropout_mlp_probability),
                Linear(embedding_size, embedding_size),
                BatchNorm1d(embedding_size),
                ReLU(),
                Dropout(self.dropout_mlp_probability),
                Linear(embedding_size, classes)).float()
        else:
            self.classification = Sequential(
                Linear(embedding_size, embedding_size),
                BatchNorm1d(embedding_size),
                ReLU(),
                Linear(embedding_size, embedding_size),
                BatchNorm1d(embedding_size),
                ReLU(),
                Linear(embedding_size, classes)).float()

    def evaluate(self, input_dict):
        """
        Calculate the metric for the dataset.

        :param input_dict:  The dictionary with 'y_true' and 'y_pred' to calculate metrics.
        :return:            The computed metric.
        """
        return self.evaluator.eval(input_dict)

    def initial_layer(self, x, edge_index):
        # Do the normalization
        x = self.input_norm(x.float())

        # Initial GNN
        x = self.relu(self.input_gnn(x, edge_index))
        return x

    def middle_layer(self, x, edge_index, layer, index):
        # Do the normalization
        x = self.normalization_layers[index](x)

        # GNN layer
        x = self.relu(layer(x, edge_index))

        return x

    def forward(self, x, edge_index, batch):
        x = self.initial_layer(x, edge_index)

        # Pass through the middle layers
        for index, layer in enumerate(self.layers):
            x = self.middle_layer(x, edge_index, layer, index=index)

        # Obtain the output
        return self.classification(global_mean_pool(x, batch))

    def training_step(self, batch, batch_idx):

        # Obtain the data from the dataset
        x = batch.x
        edge_index = batch.edge_index
        graph_batch = batch.batch

        classified = self.forward(x, edge_index, graph_batch)

        # Compare the results
        compare = batch.y

        # Squeeze the dimension
        classified = classified.squeeze(1)
        compare = compare.float()

        # Calculate the loss
        loss = self.loss(classified, compare)

        # Calculate the metric
        input_dict = {
            'y_true': compare,
            'y_pred': classified
        }

        metric = self.evaluate(input_dict)
        self.log('train', metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Obtain the data from the dataset
        x = batch.x
        edge_index = batch.edge_index
        graph_batch = batch.batch

        classified = self.forward(x, edge_index, graph_batch)

        # Compare the results
        compare = batch.y

        # Squeeze the dimension
        classified = classified.squeeze(1)
        compare = compare.float()

        # Calculate the loss
        loss = self.loss(classified, compare)

        # Calculate the metric
        input_dict = {
            'y_true': compare,
            'y_pred': classified,
            'val_loss': loss.mean().item()
        }

        return input_dict

    def validation_epoch_end(self, val_step_outputs):
        y_true_values = []
        y_pred_values = []
        val_loss_values = []

        # Mix the tensors
        for input_dict in val_step_outputs:
            y_true_values.append(input_dict['y_true'])
            y_pred_values.append(input_dict['y_pred'])
            val_loss_values.append(input_dict['val_loss'])

        # Generate tensors for evaluation
        y_true = torch.cat(y_true_values, dim=0)
        y_pred = torch.cat(y_pred_values, dim=0)

        # Calculate the dictionary
        input_dict = {
            'y_true': y_true,
            'y_pred': y_pred
        }

        metric = self.evaluate(input_dict)

        self.log('val', metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', np.mean(val_loss_values), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(list(self.parameters()), self.lr, weight_decay=self.weight_decay)

        return optimizer


