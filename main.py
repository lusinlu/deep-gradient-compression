from data_loader import *
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.utils.data as data
from math import ceil
from torch.autograd import Variable
from torch.multiprocessing import Process
import argparse
from deep_gradient_compression import DGC
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='path where to download, or from where to read data')
parser.add_argument('--momentum', default=0.6, type=int, help='momentum correlation for accumulated gradients')
parser.add_argument('--lr', default=1e-2, type=int, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='batch size which will be divided to number of model instances')
parser.add_argument('--world_size', default=2, type=int, help='number of model instances to be run parallel')
parser.add_argument('--persentages', default=[25, 6.25, 1.5625, 0.4, 0.1], type=list, help='exponential decreasing persentages of the gradients for top k selection')
parser.add_argument('--iters', default=[0, 50, 100, 200, 300], type=list, help='iterations at which persentage will be decreased (for args.persentages)')

args = parser.parse_args()


class Net_CIFAR(nn.Module):
    """Dummy Convolutional network for CIFAR10 classification"""
    def __init__(self):
        super(Net_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset():
    """Load (or download) dataset and divide the data into partitions to feed into different branches
    :return
    train_set : function,  partition of the dataset depending on rank of the process
    b_size : int, batch size at each node
    """
    dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                     download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    size = dist.get_world_size()
    # split batch size in two equal parts
    b_size = int(args.batch_size / float(size))

    # partition dataset to the number of parallel instances
    partition_size = [1. / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_size)
    partition = partition.use(dist.get_rank())
    train_set = data.DataLoader(partition, batch_size=b_size, shuffle=True)

    return train_set, b_size


def run(rank, world_size):
    """main training function"""
    device_id = rank
    train_set, b_size = partition_dataset()
    model = Net_CIFAR().cuda(device_id)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    num_batches = ceil(len(train_set.dataset) / float(b_size))

    dgc_trainer = DGC(model=model, rank=rank, size=world_size, device_id=device_id,
                      momentum=args.momentum,  full_update_layers=[4],persentages=args.persentages, itreations=args.iters)


    for epoch in range(args.epoch):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_set):
            it = epoch * len(train_set) + batch_idx

            data, target = Variable(data.cuda(device_id)), Variable(target.cuda(device_id))
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            dgc_trainer.gradient_update(it)

            optimizer.step()

        print('Rank ',dist.get_rank(), ', epoch ', epoch, ': ',epoch_loss / num_batches)


def init_processing(rank, size, fn, backend='gloo'):
    """initiale each process by indicate where the master node is located(by ip and port) and run main function
    :parameter
    rank : int , rank of current process
    size : int, overall number of processes
    fn : function, function to run at each node
    backend : string, name of the backend for distributed operations
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == '__main__':
    processes = []
    for rank in range(args.world_size):
        p = Process(target=init_processing, args=(rank, args.world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
