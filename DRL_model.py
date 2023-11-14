import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMActorCritic(nn.Module):
    def __init__(self, input_size, conv_channels, lstm_hidden_size, num_layers, output_size):
        super(LSTMActorCritic, self).__init__()
        self.drop_layer = nn.Dropout(p=0.05)
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2) # 1D Convolutional Layer
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(conv_channels, lstm_hidden_size, num_layers, batch_first=True)  # LSTM Layer
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 5),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # Transpose input for Conv1D layer (batch_size, input_size, sequence_length)
        x = self.conv1d(x)  
        x = x.permute(0, 2, 1) # Transpose back to (batch_size, sequence_length, conv_channels)
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm_hidden_size).to(device)
        out, (hidden_state, cell_state) = self.lstm(x, (h0, c0))
        state = self.drop_layer(out[:, -1, :])
        policy_dist = self.actor(state)
        value_est = self.critic(state)
        return policy_dist, value_est, state