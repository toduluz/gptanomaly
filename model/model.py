import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Building an linear encoder with Linear
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//8, self.hidden_size//16),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//16, self.hidden_size//32),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//32, self.hidden_size//64)
        )
         
        # Building an linear decoder with Linear
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size//64, self.hidden_size//32),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//32, self.hidden_size//16),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//16, self.hidden_size//8),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//8, self.hidden_size//4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.LeakyReLU(),
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
        self.mlp = nn.Linear(hidden_size,2)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        features_encoded,_ = self.lstm(inputs)
        # get last ouput of lstm encoder
        features_encoded = features_encoded[:,-1,:]
        out = self.mlp(features_encoded)
        logits = self.activation(out)
        probs = self.softmax(logits)
        return logits, probs