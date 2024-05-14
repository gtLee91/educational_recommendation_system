import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import learn2learn as l2l
from data_pre_process import df
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from MAML_model import MAMLModel
import os
from sqlalchemy import create_engine
import pandas as pd
from merge_data import cs_data
from data_pre_process import pre_process
import matplotlib.pyplot as plt

input_datas = df.drop(columns=['rating','cs_title','aimscore'])

item_datas = df['item']
output_datas = df['rating']

user_ids = np.unique(input_datas['user'])

tasks = []
for user_id in user_ids:
    user_data = input_datas[input_datas['user'] == user_id].drop(columns=['user', 'item'])
    user_item = item_datas[input_datas['user'] == user_id]
    user_ratings = output_datas[input_datas['user'] == user_id]

    train_data, test_data, train_ratings, test_ratings = train_test_split(
        user_data, user_ratings, test_size=0.3, random_state=False)
    
    train_item, test_item = train_test_split( user_item, test_size=0.3, random_state=False )

    task_data = {'train_data': train_data, 'train_ratings': train_ratings, 'train_item' : train_item,
                 'test_data': test_data, 'test_ratings': test_ratings, 'test_item' : test_item}
    tasks.append(task_data)

#print(tasks[0])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32) 
        self.targets = torch.tensor(targets, dtype=torch.float32)  # PyTorch 텐서로 변환

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index].unsqueeze(0) 



def compute_loss(model, inputs, targets):
    inputs = inputs.float() 
    predictions = model(inputs)
    #print(predictions)
    loss = F.mse_loss(predictions, targets)
    #print(loss)
    return loss


lr = 0.0002
num_tasks = len(tasks)
adaptation_steps = 10 
num_epochs = 40  

input_dim = tasks[0]['train_data'].shape[1]
output_dim = 1  
model = MAMLModel(input_dim, output_dim)
maml = l2l.algorithms.MAML(model, lr=lr, first_order=False)
optimizer = torch.optim.Adam(maml.parameters(), lr=lr)

# Lists to store metrics for each epoch
train_losses = []
test_mses = []
test_maes = []

# Train loop
for epoch in range(num_epochs):
    train_loss_epoch = 0.0
    test_loss_epoch = 0.0
    test_mse_epoch = 0.0
    test_mae_epoch = 0.0
    
    for task_data in tasks:
        learner = maml.clone()
        train_data = task_data['train_data'].values
        train_ratings = task_data['train_ratings'].values
        test_data = task_data['test_data'].values
        test_ratings = task_data['test_ratings'].values

        train_dataset = CustomDataset(train_data, train_ratings)
        test_dataset = CustomDataset(test_data, test_ratings)

        train_batch_size = min(len(train_dataset), 32)
        test_batch_size = min(len(test_dataset), 32)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)

        for step in range(adaptation_steps):
            for inputs, targets in train_loader:
                train_loss = compute_loss(learner, inputs, targets)
                learner.adapt(train_loss)

        for inputs, targets in test_loader:
            test_loss = compute_loss(learner, inputs, targets)
            test_mse = F.mse_loss(learner(inputs), targets)
            test_mae = F.l1_loss(learner(inputs), targets)
            optimizer.zero_grad()
            test_loss.backward()
            optimizer.step()

        train_loss_epoch += train_loss.item()
        test_loss_epoch += test_loss.item()
        test_mse_epoch += test_mse.item()
        test_mae_epoch += test_mae.item()

    # Compute average losses
    train_loss_epoch /= len(tasks)
    test_loss_epoch /= len(tasks)
    test_mse_epoch /= len(tasks)
    test_mae_epoch /= len(tasks)

    # Append metrics to the lists
    train_losses.append(train_loss_epoch)
    test_mses.append(test_mse_epoch)
    test_maes.append(test_mae_epoch)
    
    # Print metrics for the epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_epoch:.4f}, '
          f'Test Loss: {test_loss_epoch:.4f}, Test MSE: {test_mse_epoch:.4f}, Test MAE: {test_mae_epoch:.4f}')


# Plotting the metrics
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))

plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_mses, label='Test MSE')
plt.plot(epochs, test_maes, label='Test MAE')

plt.xlabel('Epochs')
plt.ylabel('Loss/MSE/MAE')
plt.title('Training and Evaluation Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.show()

db_uri = "mysql+mysqlconnector://root:rhks25714334@localhost:3306/recommend_sys"
engine = create_engine(db_uri)

user_num = 1
query_profile = f"SELECT UserID, Gender, Age, PreferredCategory, PreferredTopic, Level, PreferredStyle, AimScore FROM recommend_sys.user_profile WHERE UserID = {user_num}"
user_profile = pd.read_sql(query_profile, engine)

user_profile.rename(columns={'UserID': 'user', 'PreferredCategory': 'pf_category', 'PreferredTopic': 'pf_topic', 
                              'Level': 'pf_level', 'PreferredStyle': 'pf_style', 'AimScore': 'aimscore', 
                              'Gender': 'gender', 'Age': 'age'}, inplace=True)

new_user = pd.DataFrame(user_profile)
new_user_cp = pd.concat([new_user] * 322, ignore_index=True)
new_user_df = pd.concat([new_user_cp, cs_data], axis=1)
new_user_in = pre_process(new_user_df)

#print(new_user_in)
# 모델을 사용하여 새로운 사용자의 강의 평점 예측
test_maml_user_data = new_user_in.drop(columns=['user', 'item','cs_title','gender','age','aimscore'])
test_maml_item_data = new_user_in['item']

def recommend_items(model, user_data, user_item, top_n=20):
    user_tensor = torch.tensor(user_data.values, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(user_tensor).flatten()
    
    predicted_ratings = list(zip(user_item, predictions.numpy()))
    
    recommended_items = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    
    return recommended_items


recommended_items = recommend_items(model, test_maml_user_data, test_maml_item_data, top_n=20)
print("recommend item:", recommended_items)

#---------------------------------------
folder_path = 'trained_model'
#model save
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': maml.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
},  os.path.join(folder_path, 'MAML_model_3.pth'))
