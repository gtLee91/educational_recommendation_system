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
from MAML_model import MAMLModel
from torch.utils.data import DataLoader, TensorDataset

new_user_data = {
    'user': [501],
    'pf_category': ['pte'],
    'pf_topic': ['writing,speaking,listening'],
    'pf_level': ['beginner'],
    'pf_style': ['example'],
    'aimscore': [6],
    'gender': ['Female'],
    'age': [35]
}

num_rows = cs_data.shape[0]
new_user = pd.DataFrame(new_user_data)

new_user_cp = pd.concat([new_user] * num_rows, ignore_index=True)

new_user_df = pd.concat([new_user_cp, cs_data], axis=1)

new_user_in = pre_process(new_user_df)

new_column_order = ['user', 'item', 'cs_title', 'cs_style', 
                    'cs_level', 'pf_style', 'pf_level',
                    'cs_category_ielts', 'cs_category_pte', 'cs_topic_writing', 
                    'cs_topic_speaking', 'cs_topic_reading', 'cs_topic_listening', 
                    'cs_topic_vocabulary', 'cs_topic_grammar', 'pf_category_ielts', 
                    'pf_category_pte', 'pf_topic_writing', 'pf_topic_speaking', 
                    'pf_topic_reading', 'pf_topic_listening', 'pf_topic_vocabulary', 'pf_topic_grammar']


new_user_in = new_user_in.reindex(columns=new_column_order)
#print(new_user_in.head(10))

test_maml_user_data = new_user_in.drop(columns=['user', 'item','cs_title'])

test_maml_item_data = new_user_in['item']


sequence_length = 3
test_lstm_seq_data = seq_dataset(new_user_in, sequence_length)
#print(test_lstm_seq_data[0])

batch_size = 128
test_lstm_data_loader = DataLoader(test_lstm_seq_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 20
conv_channels = 128
hidden_size = 128
num_layers = 1
output_size = 1

t_model = LSTMActorCritic(input_size, conv_channels, hidden_size, num_layers, output_size).to(device)
t_optimizer = optim.Adam(t_model.parameters(), lr=0.0002)

checkpoint = torch.load('trained_model/drl_model_6.pth')
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

        next_state, item_list, action_list, exploration_reward, done = env.step(action)

        print("item_list : ",item_list)
        print("action_list : ",action_list)

        all_item_list.extend(item_list)
        all_rating_list.extend(action_list)

        if done == 1:
            last_num = policy_dist.size(0)

        if done == 0:
            state = next_state
            env.reset()
    
    result_data = cs_data[cs_data['item'].isin(all_item_list)]

    print(result_data)

MAML_checkpoint = torch.load('trained_model/MAML_model_3.pth')
MAML_model = MAMLModel(20, 1)
MAML_model.load_state_dict(MAML_checkpoint['model_state_dict'])

MAML_optimizer = optim.Adam(MAML_model.parameters(), lr=0.0002)
MAML_optimizer.load_state_dict(MAML_checkpoint['optimizer_state_dict'])

def recommend_items(model, user_data, user_item, top_n=20):
    user_data = user_data.astype(float)
    
    user_tensor = torch.tensor(user_data.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(user_tensor).flatten()

    predicted_ratings = list(zip(user_item, predictions.numpy()))
    
    recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    
    return recommended_items



recommended_items = recommend_items(MAML_model, test_maml_user_data, test_maml_item_data, top_n=20)
print(recommended_items)

recommended_item_ids = [item[0] for item in recommended_items]

common_items = set(all_item_list) & set(recommended_item_ids)


common_item_list = list(common_items)
print("common_item ID:", common_item_list)

if common_item_list:
    common_item_details = cs_data[cs_data['item'].isin(common_item_list)]
    print(common_item_details)
else:
    print("No common item")

