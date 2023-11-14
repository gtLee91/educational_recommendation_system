import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import os
import pandas as pd
from merge_data import cs_data
from data_pre_process import pre_process
from testing_env import Environment
from testing_sq_dataset import seq_dataset
from DRL_model import LSTMActorCritic
from torch.utils.data import DataLoader, TensorDataset

new_user_data = {
    'user': [5001],
    'pf_category': ['ielts'],
    'pf_topic': ['writing,speaking,vocabulary'],
    'pf_level': ['advanced'],
    'pf_style': ['example'],
    'aimscore': [6],
    'gender': ['Female'],
    'age': [35]
}

new_user = pd.DataFrame(new_user_data)

new_user_cp = pd.concat([new_user] * 322, ignore_index=True)

new_user_df = pd.concat([new_user_cp, cs_data], axis=1)

new_user_in = pre_process(new_user_df)
new_column_order = ['user', 'item', 'cs_title', 'cs_style', 
                    'cs_level', 'pf_style', 'pf_level', 'aimscore', 'age', 'gender', 
                    'cs_category_ielts', 'cs_category_pte', 'cs_topic_writing', 
                    'cs_topic_speaking', 'cs_topic_reading', 'cs_topic_listening', 
                    'cs_topic_vocabulary', 'cs_topic_grammar', 'pf_category_ielts', 
                    'pf_category_pte', 'pf_topic_writing', 'pf_topic_speaking', 
                    'pf_topic_reading', 'pf_topic_listening', 'pf_topic_vocabulary', 'pf_topic_grammar']


new_user_in = new_user_in.reindex(columns=new_column_order)
print(new_user_in.head(10))
sequence_length = 3
test_lstm_seq_data = seq_dataset(new_user_in, sequence_length)

batch_size = 128
test_lstm_data_loader = DataLoader(test_lstm_seq_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 23
conv_channels = 128
hidden_size = 128
num_layers = 1
output_size = 1

t_model = LSTMActorCritic(input_size, conv_channels, hidden_size, num_layers, output_size).to(device)
t_optimizer = optim.Adam(t_model.parameters(), lr=0.00002)

checkpoint = torch.load('trained_model/drl_model.pth')
t_model.load_state_dict(checkpoint['drl_model_state_dict'])
t_optimizer.load_state_dict(checkpoint['drl_optimizer_state_dict'])

t_model.eval()
all_item_list = []
all_rating_list = []  
with torch.no_grad():
    env = Environment(test_lstm_data_loader)
    done = env.done
    state = env.state
    while not done:

        policy_dist, value_est, st = t_model(state)

        action = torch.argmax(policy_dist, dim=1)

        next_state, item_list, action_list, done = env.step(action)

        all_item_list.extend(item_list)
        all_rating_list.extend(action_list)

        if done == 1:
            last_num = policy_dist.size(0)

        if done == 0:
            state = next_state
    
    print(f'item_list:{all_item_list}, rating_list:{all_rating_list}')
    result_data = cs_data[cs_data['item'].isin(all_item_list)]

    
    print(result_data)

