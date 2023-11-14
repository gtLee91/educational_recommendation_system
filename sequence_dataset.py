import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_pre_process import df

def seq_dataset(df, sequence_length):
    sequences = []
    labels = []
    users = []
    items = []

    for i in range(len(df) - (sequence_length-1)):
        sequence = df[i:i+sequence_length].drop(['user', 'item', 'cs_title', 'rating'], axis=1).values
        label = df.iloc[i+sequence_length-1]['rating']
        user = df.iloc[i+sequence_length-1]['user']
        item = df.iloc[i+sequence_length-1]['item']
        
        sequences.append(sequence)
        labels.append(label-1)
        users.append(user)
        items.append(item)

    sequences = torch.tensor(sequences, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    users = torch.tensor(users, dtype=torch.int32)
    items = torch.tensor(items, dtype=torch.int32)
    
    dataset = TensorDataset(sequences, labels, users, items)
    
    return dataset

user_df_train, user_df_test = train_test_split(df, test_size=0.3, random_state=0, shuffle=False)