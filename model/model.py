import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    #Multi layers (max 5) autoencoder with dropout 0.1 and LeakyRELU() activation
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//8, hidden_dim//16),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//16, hidden_dim//8),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//8, hidden_dim//4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


        

class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim = 768, n_hidden = 768):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True, dropout=0.1, batch_first=True, num_layers=10)
        self.out = nn.Linear(n_hidden * 2, n_hidden)
        # self.ae = AutoEncoder(n_hidden, n_hidden//2, n_hidden)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state, n_hidden = 768):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X, n_hidden = 768):
        # input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        # input = X.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        # hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # cell_state = Variable(torch.zeros(1*2, len(X), n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(X)
        # output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state[-2:], n_hidden)
        return self.out(attn_output) # model : [batch_size, num_classes], attention : [batch_size, n_step]
        # Normalize
        # ae_input = F.normalize(ae_input, p=2, dim=1)
        # ae_output = self.ae(ae_input)
        # return ae_input, ae_output


