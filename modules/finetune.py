import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import modules.data as dataset

class FineTuner:
    def __init__(self, args, model):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        # Data Acquisition
        get_dataset = {
            "mit_bih": dataset.get_mit_bih,
            "cifar10": dataset.get_cifar10

        }[args.data_type.lower()]
        train_dataset, test_dataset = get_dataset()

        #how to load load ECG (in process)
        from scipy.misc import electrocardiogram
        ecg = electrocardiogram()
        ecg.shape
        import matplotlib.pyplot as plt
        fs = 360
        time = np.arange(ecg.size) / fs
        plt.plot(time, ecg)
        plt.xlabel("time in s")
        plt.ylabel("ECG in mV")
        plt.xlim(9, 10.2)
        plt.ylim(-1, 1.5)
        plt.show()
        ##########################33

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=args.test_batch_size,
                                                       shuffle=False, **kwargs)

        self.args = args
        self.model = model

        self.criterion = nn.CrossEntropyLoss()

        # self.prunner = FilterPrunner(self.model, args)
        self.model.train()

    def train_epoch(self, optimizer=None, rank_filters = False):
        for batch_idx, (data, label) in enumerate(self.train_loader):
            if self.args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            self.train_batch(optimizer, batch_idx, data, label, rank_filters)

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters):
        self.model.zero_grad()

        if rank_filters:
            print("fill in the code")

        else:
            loss = self.criterion(self.model(batch), label)
            loss.backward()
            optimizer.step()
            print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        for data, label in self.test_loader:
            if self.args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            output = self.model(data) #forward
            y = output
            test_loss += self.criterion(y, label).item()
            # get the index of the max log-probability
            pred = y.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))