import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Environment:
    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.data = next(self.dataloader)

        self.state = self.data[0].to(device)
        self.rating = self.data[1]
        self.user = self.data[2]
        self.item = self.data[3]
                
        self.done = 0
    def reset(self):
        self.rating = self.next_data[1]
    def step(self, action):         
        accuracy_count = 0
        reward = self.rating
        
        for i in range(len(action)):
            if action[i] == reward[i]:
                accuracy_count += 1

        try:
            self.next_data = next(self.dataloader)
        except StopIteration:
            next_state = 0
            self.done = 1
        else:
            next_state = self.next_data[0]
            self.done = 0
        
        return next_state, reward, accuracy_count, self.done