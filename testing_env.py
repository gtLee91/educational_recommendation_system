import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Environment:
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.data = next(self.dataloader)

        self.state = self.data[0].to(device)
        self.user = self.data[1]
        self.item = self.data[2]
                
        self.done = 0
        self.item_count = {} 
        
        for item_id in self.item:
            self.item_count[item_id.item()] = self.item_count.get(item_id.item(), 0) + 1
    def reset(self):
        self.user = self.next_data[1]
        self.item = self.next_data[2]
        
        for item_id in self.item:
            self.item_count[item_id.item()] = self.item_count.get(item_id.item(), 0) + 1
    def step(self, action):         
        item_list = []
        action_list = []

        for i in range(len(action)):
            if action[i] >= 3:
                item_list.append(self.item[i].item())
                action_list.append(action[i].item()+1)

        exploration_bonus = 0.1
        exploration_reward = torch.zeros_like(self.item, dtype=torch.float32)

        for i, item_id in enumerate(self.item):
            item_id = item_id.item()
            
            if self.item_count[item_id] < 5:
                exploration_reward[i] = exploration_bonus * (5 - self.item_count[item_id])

        try:
            self.next_data = next(self.dataloader)
        except StopIteration:
            next_state = 0
            self.done = 1
        else:
            next_state = self.next_data[0]
            self.done = 0
        
        return next_state, item_list, action_list, exploration_reward, self.done