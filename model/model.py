import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Building an linear encoder with Linear
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//8, self.hidden_size//16),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//16, self.hidden_size//32),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//32, self.hidden_size//64)
        )
         
        # Building an linear decoder with Linear
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size//64, self.hidden_size//32),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//32, self.hidden_size//16),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//16, self.hidden_size//8),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//8, self.hidden_size//4),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size//2, self.hidden_size)
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LSTMDiscriminator(nn.Module):
    def __init__(self,hidden_size: int = 384):

        super().__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.mlp = nn.Linear(hidden_size,1)
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        features_encoded,_ = self.lstm(inputs)
        # get last ouput of lstm encoder
        features_encoded = features_encoded[:,-1,:]
        out = self.mlp(features_encoded)
        out = self.activation(out)
        return out