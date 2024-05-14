import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MAML model define
class MAMLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MAMLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  
        self.fc2 = nn.Linear(32, 16)  
        self.fc3 = nn.Linear(16, output_dim)  

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x