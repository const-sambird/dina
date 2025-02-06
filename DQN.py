import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, layer_features):
        super(DQN, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(n_observations, layer_features[0]))
        for i in range(1, len(layer_features)):
            self.layers.append(nn.Linear(layer_features[i-1], layer_features[i]))
        self.layers.append(nn.Linear(layer_features[len(layer_features)-1], n_actions))
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            return x
