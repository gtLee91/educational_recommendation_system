import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import os
import pandas as pd
import importlib
from sqlalchemy import create_engine
#from merge_data import cs_data
from data_pre_process import pre_process
from testing_env import Environment
from testing_sq_dataset import seq_dataset
from DRL_model import LSTMActorCritic
from MAML_model import MAMLModel
from torch.utils.data import DataLoader, TensorDataset

def recommend_items(model, user_data, user_item, top_n=20):
    
    user_tensor = torch.tensor(user_data.values, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(user_tensor).flatten()
    
    predicted_ratings = list(zip(user_item, predictions.numpy()))
    
    recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    
    return recommended_items

def recommend_result(session):
    import merge_data

    importlib.reload(merge_data)
    cs_data = merge_data.cs_data

    db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
    engine = create_engine(db_uri)

    result_data = pd.DataFrame()

    user_num = session.get('user_num')

    query_profile = f"SELECT UserID, Gender, Age, PreferredCategory, PreferredTopic, Level, PreferredStyle, AimScore FROM recommend_sys.user_profile WHERE UserID = {user_num}"
    user_profile = pd.read_sql(query_profile, engine)

    user_profile.rename(columns={'UserID': 'user'}, inplace=True)
    user_profile.rename(columns={'PreferredCategory': 'pf_category'}, inplace=True)
    user_profile.rename(columns={'PreferredTopic': 'pf_topic'}, inplace=True)
    user_profile.rename(columns={'Level': 'pf_level'}, inplace=True)
    user_profile.rename(columns={'PreferredStyle': 'pf_style'}, inplace=True)
    user_profile.rename(columns={'AimScore': 'aimscore'}, inplace=True)
    user_profile.rename(columns={'Gender': 'gender'}, inplace=True)
    user_profile.rename(columns={'Age': 'age'}, inplace=True)

    num_rows = cs_data.shape[0]
    print(num_rows)

    new_user = pd.DataFrame(user_profile)
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
    
    test_maml_user_data = new_user_in.drop(columns=['user', 'item','cs_title'])
    test_maml_item_data = new_user_in['item']

    sequence_length = 3
    test_lstm_seq_data = seq_dataset(new_user_in, sequence_length)

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

            all_item_list.extend(item_list)
            all_rating_list.extend(action_list)

            if done == 1:
                last_num = policy_dist.size(0)

            if done == 0:
                state = next_state
                env.reset()
        
        result = cs_data[cs_data['item'].isin(all_item_list)]
        print(result)

    
    MAML_checkpoint = torch.load('trained_model/MAML_model_3.pth')
    MAML_model = MAMLModel(20, 1)
    MAML_model.load_state_dict(MAML_checkpoint['model_state_dict'])

    MAML_optimizer = optim.Adam(MAML_model.parameters(), lr=0.0002)
    MAML_optimizer.load_state_dict(MAML_checkpoint['optimizer_state_dict'])
    MAML_model.eval()

    recommended_items = recommend_items(MAML_model, test_maml_user_data, test_maml_item_data, top_n=20)
    recommended_item_ids = [item[0] for item in recommended_items]
    print(recommended_item_ids)
    common_items = set(all_item_list) & set(recommended_item_ids)
    common_item_list = list(common_items)


    if common_item_list:
        result_data = cs_data[cs_data['item'].isin(common_item_list)]
        print(result_data)
    else:
        print("No items")

    return result_data
