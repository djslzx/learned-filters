from random import random

import torch as T
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, n):
        super(Net, self).__init__()
        hidden_size = (n+1)//2  # Use avg of 2 layer sizes
        self.w1 = nn.Linear(n, hidden_size)  
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = T.sigmoid(self.w1(x))
        x = T.sigmoid(self.w2(x))
        return x

def train(net, xs, ys, epochs):
    criterion = nn.BCELoss()
    optimizer = T.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    dataloader = T.utils.data.TensorDataset(xs, ys)

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 1):
            x, y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    random_data = T.rand([1,5])
    net = Net(n=5)
    result = net(random_data)
    print(result)
