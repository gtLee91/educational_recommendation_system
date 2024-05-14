import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import os
from environment import Environment
from sequence_dataset import seq_dataset, user_df_train
from DRL_model import LSTMActorCritic
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sequence_length = 3
seq_data = seq_dataset(user_df_train, sequence_length)

batch_size = 128
data_loader = DataLoader(seq_data, batch_size=batch_size, shuffle=False)

input_size = 20
conv_channels = 128
hidden_size = 128
num_layers = 1
output_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMActorCritic(input_size, conv_channels, hidden_size, num_layers, output_size).to(device)
critic_criterion = nn.MSELoss()
actor_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# training loop 
critic_losses = []
actor_losses = []
accuracies = []
num_epochs = 50
for epoch in tqdm.tqdm(range(num_epochs)):
    env = Environment(data_loader)
    total_critic_loss = 0
    total_actor_loss = 0
    total_count = 0
    done = env.done
    state = env.state

    while not done:
        optimizer.zero_grad()

        policy_dist, value_est, st = model(state)

        action = torch.multinomial(policy_dist, 1)
        next_state, reward, exploration_reward, count, done = env.step(action)
        
        actor_loss = actor_criterion(policy_dist, reward.long())
        combined_reward = reward + exploration_reward
        critic_loss = critic_criterion(value_est, combined_reward.unsqueeze(1))

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        
        optimizer.step()
    
        total_critic_loss += critic_loss.item()
        total_actor_loss += actor_loss.item()
        total_count += count

        if done == 1:
            last_num = policy_dist.size(0)
        if done == 0:
            state = next_state
            env.reset()
    
    average_critic_loss = total_critic_loss / len(data_loader)
    average_actor_loss = total_actor_loss / len(data_loader)
    accuracy = total_count / (((len(data_loader)-1) * 128) + last_num)
    critic_losses.append(average_critic_loss)
    actor_losses.append(average_actor_loss)
    accuracies.append(accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], critic_Loss: {average_critic_loss:.4f}, actor_loss:{average_actor_loss}, accuracy:{accuracy}')


folder_path = 'trained_model'
#model save
torch.save({
    'drl_model_state_dict': model.state_dict(),
    'drl_optimizer_state_dict': optimizer.state_dict(),
},  os.path.join(folder_path, 'drl_model_7.pth'))

# critic loss
plt.figure()
plt.plot(range(num_epochs), critic_losses)
plt.xlabel('Epoch')
plt.ylabel('Critic Loss')
plt.title('Critic Loss over Epochs')

# Actor Loss 
plt.figure()
plt.plot(range(num_epochs), actor_losses)
plt.xlabel('Epoch')
plt.ylabel('Actor Loss')
plt.title('Actor Loss over Epochs')

# accuracy
plt.figure()
plt.plot(range(num_epochs), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')

plt.show()