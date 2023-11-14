import torch
from torch.utils.data import DataLoader, TensorDataset

def seq_dataset(df, sequence_length):
    sequences = []
    users = []
    items = []

    for i in range(len(df) - (sequence_length-1)):
        sequence = df[i:i+sequence_length].drop(['user', 'item', 'cs_title'], axis=1).values
        user = df.iloc[i+sequence_length-1]['user']
        item = df.iloc[i+sequence_length-1]['item']
        
        sequences.append(sequence)
        users.append(user)
        items.append(item)

    sequences = torch.tensor(sequences, dtype=torch.float32)
    users = torch.tensor(users, dtype=torch.int32)
    items = torch.tensor(items, dtype=torch.int32)
    
    dataset = TensorDataset(sequences, users, items)
    
    return dataset
