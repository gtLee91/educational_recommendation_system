import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import os
from environment import Environment
from sequence_dataset import seq_dataset, user_df_test
from DRL_model import LSTMActorCritic
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

sequence_length = 3
test_lstm_seq_data = seq_dataset(user_df_test, sequence_length)
num_data_rows = user_df_test.shape[0]
print("The number of data rows in user_df_test:", num_data_rows)

batch_size = 128
test_lstm_data_loader = DataLoader(test_lstm_seq_data, batch_size=batch_size, shuffle=False)

num_batches = len(test_lstm_data_loader)
print("The number of batches in test_lstm_data_loader:", num_batches)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#input_size = 23
input_size = 20
conv_channels = 128
hidden_size = 128
num_layers = 1
output_size = 1

t_model = LSTMActorCritic(input_size, conv_channels, hidden_size, num_layers, output_size).to(device)
t_optimizer = optim.Adam(t_model.parameters(), lr=0.0002)

checkpoint = torch.load('trained_model/drl_model_7.pth')
t_model.load_state_dict(checkpoint['drl_model_state_dict'])
t_optimizer.load_state_dict(checkpoint['drl_optimizer_state_dict'])

total_count = 0
total_size = 0
true_labels = []
predicted_labels = []
batch_acc = []
batch_rmse = []
batch_mae = []
batch_f1score = []
check_batch_num = 0


# model evaluation
t_model.eval()  
with torch.no_grad():
    env = Environment(test_lstm_data_loader)
    done = env.done
    state = env.state
    while not done:
        check_batch_num += 1
        policy_dist, value_est, st = t_model(state)

        action = torch.argmax(policy_dist, dim=1)

        next_state, reward, exploration_reward, count, done = env.step(action)
        total_count += count
        total_size += policy_dist.size(0)
        
        true_labels.extend(reward.cpu().numpy())
        predicted_labels.extend(action.cpu().numpy())

        last_num = policy_dist.size(0)

        if done == 1:  
            weight = (last_num / 128)
        else:  
            weight = 1.0  

        batch_accuracy = count / policy_dist.size(0)
        batch_RMSE = mean_squared_error(reward, action, squared=False)
        batch_MAE = mean_absolute_error(reward, action)
        batch_F1 = f1_score(reward, action, average='micro')

        batch_acc.append(batch_accuracy)
        batch_rmse.append(batch_RMSE * weight)
        batch_mae.append(batch_MAE * weight)
        batch_f1score.append(batch_F1)

        if done == 1:
            last_num = policy_dist.size(0)
            print(f'last_num:{last_num}')
        if done == 0:
            env.reset()
            state = next_state
            
    # accuracy
    eval_accuracy = total_count / (((len(test_lstm_data_loader)-1)*128)+last_num)
    # RMSE
    #rmse = mean_squared_error(true_labels, predicted_labels, squared=False)
    rmse = sum(batch_rmse) / len(test_lstm_data_loader)
    # MAE
    #mae = mean_absolute_error(true_labels, predicted_labels)
    mae = sum(batch_mae) / len(test_lstm_data_loader)
    # F1 score
    f1 = f1_score(true_labels, predicted_labels, average='micro')

    print(f'Total accuracy: {eval_accuracy:.4f}, Total RMSE: {rmse:.4f}, Total MAE: {mae:.4f}, Total F1 score: {f1:.4f}')

# accuracy
plt.figure()
plt.plot(range(check_batch_num), batch_acc)
plt.xlabel('batch num')
plt.ylabel('Accuracy')
plt.title('Accuracy over Batch number')
plt.ylim(0.5, 1.5)

# RMSE
plt.figure()
plt.plot(range(check_batch_num), batch_rmse)
plt.xlabel('batch num')
plt.ylabel('RMSE')
plt.title('RMSE over Batch number')
plt.ylim(-1, 1)

# MAE
plt.figure()
plt.plot(range(check_batch_num), batch_mae)
plt.xlabel('batch num')
plt.ylabel('MAE')
plt.title('MAE over Batch number')
plt.ylim(-0.3, 0.3)

# F1 score
plt.figure()
plt.plot(range(check_batch_num), batch_f1score)
plt.xlabel('batch num')
plt.ylabel('F1 Score')
plt.title('F1 Score over Batch number')
plt.ylim(0.5, 1.5)

plt.show()
