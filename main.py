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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='path where to download, or from where to read data')
parser.add_argument('--momentum', default=0.8, type=int, help='momentum correlation for accumulated gradients')
parser.add_argument('--lr', default=1e-2, type=int, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='batch size which will be divided to number of model instances')
parser.add_argument('--world_size', default=2, type=int, help='number of model instances to be run parallel')
parser.add_argument('--threshold', default=0.2, type=float, help='threshold for large gradients, range 0. - 1.')

args = parser.parse_args()


# Simple convolutional network for classification
class Net_CIFAR(nn.Module):
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


# Load (or download) dataset and divide data for feeding in different branches
def partition_dataset():
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


# construct tensor with the same shape as weights and put large gradients
def sparse_to_dense(grad, shape):
    dense_grad = torch.zeros(shape)
    dense_grad = dense_grad.put_(grad[1], grad[0])
    return dense_grad


# obtain large gradients from all branches and avarage them
def avarage_gradient_sparce(sparse_grad):
    size = float(dist.get_world_size())
    updated_grads = []
    for sparse_grad, shape in sparse_grad:
        layer_grad = sparse_to_dense(sparse_grad, shape)
        dist.all_reduce(layer_grad, op=dist.reduce_op.SUM, group=0)
        layer_grad /= size
        updated_grads.append(layer_grad)
    return updated_grads


# separate and store "total number of gradients x threshold" amount of largest gradients
def get_large_gradients(param, small_grads, layer_id):
    current_grad = param.grad.data.view(-1) + small_grads[layer_id][0]
    _, indices = torch.sort(torch.abs(current_grad), descending=True)

    sparse_indices = indices[:int(len(param.grad.data.view(-1)) * args.threshold)]
    sparse_values = current_grad.take(sparse_indices)
    small_grads[layer_id][0][sparse_indices] = 0

    sparse_grad = [[sparse_values, sparse_indices], param.grad.data.shape]
    return sparse_grad, indices

# accumulate small gradients until they pass threshold
def accumulate_small_grads(param, sorted_indices, small_grads, layer_id):
    small_grad_indices = sorted_indices[int(len(param.grad.data.view(-1)) * args.threshold):]
    small_values = param.grad.data.take(small_grad_indices)

    tmp_tensor = torch.zeros(small_grads[layer_id][0].shape)
    tmp_tensor.put_(small_grad_indices, small_values)

    # apply momentum term to new gradients
    tmp_tensor += small_grads[layer_id][0] * args.momentum
    small_grads[layer_id][0] += tmp_tensor


def run():
    torch.manual_seed(1111)
    train_set, b_size = partition_dataset()
    model = Net_CIFAR()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    num_batches = ceil(len(train_set.dataset) / float(b_size))
    small_grads = [[torch.zeros(int(torch.prod(torch.tensor(W.shape)))), name] for name, W in model.named_parameters()]

    for epoch in range(args.epoch):
        epoch_loss = 0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss
            loss.backward()
            sparse_grad_branch = []
            # choose gradients from each layer for passing and storing
            for layer_id, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    large_grads, sorted_indices = get_large_gradients(param, small_grads, layer_id)
                    sparse_grad_branch.append(large_grads)
                    accumulate_small_grads(param, sorted_indices, small_grads, layer_id)

            # update gradients
            updated_grads = avarage_gradient_sparce(sparse_grad=sparse_grad_branch)
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    param.grad.data = updated_grads[i]
            optimizer.step()

        print('Rank ',dist.get_rank(), ', epoch ', epoch, ': ',epoch_loss / num_batches)


def init_processing(rank, size, fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn()


if __name__ == '__main__':
    processes = []
    for rank in range(args.world_size):
        p = Process(target=init_processing, args=(rank, args.world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
