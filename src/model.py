from random import random

import torch as T
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, n):
        """
        Simple Neural Net with 1 hidden layer
        """
        super(Net, self).__init__()
        self.input_size = n
        self.hidden_size = (n+1)//2  # Use avg of 2 layer sizes
        self.output_size = 1

        self.w1 = nn.Linear(self.input_size, self.hidden_size)  
        self.w2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Performs Forward Propagation
        """
        x = T.sigmoid(self.w1(x))
        x = T.sigmoid(self.w2(x))
        return x

    def train(self, xs, ys, epochs):
        """
        Train on examples for a certain number of epochs
        """
        criterion = nn.BCELoss()    # Binary cross entropy
        optimizer = T.optim.SGD(self.parameters(), 
                                lr=0.001, 
                                momentum=0.9)
        dataset = T.utils.data.TensorDataset(xs, ys)
        dataloader = T.utils.data.DataLoader(dataset, shuffle=True)

        for epoch in range(1, epochs+1):
            # running_loss = 0.0
            for i, (x,y) in enumerate(dataloader, 1):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                loss = criterion(self(x), y)
                loss.backward()
                optimizer.step()

                # # print statistics
                # running_loss += loss.item()
                # if i % 1000 == 0:
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch, i, running_loss / 2000))
                #     running_loss = 0.0

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        T.save(self.state_dict(), "Model")
        print('Finished Training')

    def __str__(self):
        return ("{} input nodes, {} hidden nodes, {} output nodes"
                .format(self.input_size, self.hidden_size, self.output_size))

class WordNet():
    
    def __init__(self, n, c):
        self.n = n
        self.c = c
        self.net = Net(n*c)

    def train(self, xs, ys, epochs):
        # Reformat xs from list of Words to 2D tensor 
        train_xs = T.stack([x.model_type for x in xs])
        # Reformat ys from list of 1/0s to column tensor
        train_ys = T.tensor(ys, dtype=T.float).reshape((-1,1))
        self.net.train(train_xs, train_ys, epochs)

    def __call__(self, x):
        return self.net(x.model_type)

    def __str__(self):
        return ("n={}, c={}, net: {}"
                .format(self.n, self.c, self.net))


if __name__ == '__main__':
    random_data = T.rand([1,5])
    net = Net(n=5)
    result = net(random_data)
    print(result)
