import torch
import torch.nn as nn

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.5):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.gru(x, h0)
        return out
    
class Classifier(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)      # Second layer

    def forward(self, x):
        x = self.fc1(x[:, -1, :])
        x = nn.ReLU()(x)               # Activation function
        out = self.fc2(x)
        return out
