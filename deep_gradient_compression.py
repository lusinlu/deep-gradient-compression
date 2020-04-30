import torch
torch.manual_seed(0)
from torch import nn
import torch.distributed as dist
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


class DGC(nn.Module):
    def __init__(self, model, rank, size,device_id, momentum, full_update_layers, persentages, itreations):
        """Class for performing sparse distributed gradient updates before backpropogation
        :parameter
        model : torch.Sequentioal, main model to be trained in data-parallel distributed manner
        rank : int, rank of the process on which class object will be allocated
        size: int, overall number of processes
        momentum : int, value of the momentum correlation
        full_update_layers : list of ints, layer indexes which will be updated without sparsification
        persentages : list of floats, persentages of sparsification
        iterations : list of ints, iterations at which persentages of sparsification will be changed
        """
        super(DGC, self).__init__()
        self.layers = {}
        self.shapes = []

        self.rank = rank
        self.size = size
        self.device_id = device_id
        self.main_model = model
        self.momentum = momentum
        self.compressed_size = None
        self.full_update_layers = full_update_layers

        self.percentages = persentages
        self.iterations = itreations
        self.current_persentage = None

        for i, (name, layer) in enumerate(model.named_parameters()):
            if layer.requires_grad and len(layer.size()) == 4 and i not in self.full_update_layers:
                self.layers[name] = torch.zeros(layer.size())
                self.shapes.append(layer.size())



    def avarage_gradient_dense(self):
        """ Gradient averaging of layers without sparsification """

        for i, (name, p) in enumerate(self.main_model.named_parameters()):
            tensor = p.grad.data.cpu()

            if i in self.full_update_layers or len(tensor.shape) != 4:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= float(self.size)


    def select_top_values_and_indices(self, tensor, name=None, momentum=None):
        """Selecting top k gradient per layer
        :parameter
        tensor : 4D tensor, tensor of gradients at the certain layer
        name : string, name of the layer
        momentum : float, value of momentum correlation
        :return
        top_indices : tensor, indices with top values at current layer
        top_values : tensor, values at top_indices
        """
        current_layer = tensor + self.layers[name]

        current_layer = current_layer.flatten()
        kbig = int(len(current_layer) * self.current_persentage / 100)
        if kbig == 0:
            kbig = 10
        _, top_indices_unsorted = torch.topk(torch.abs(current_layer), kbig)
        top_values_unsorted = torch.take(current_layer, top_indices_unsorted)

        indices_sorted = torch.argsort(top_values_unsorted)
        top_values = top_values_unsorted[indices_sorted]
        top_indices = top_indices_unsorted[indices_sorted]

        small_values_tensor = tensor.clone()
        small_values_tensor = small_values_tensor.put_(top_indices, torch.zeros(len(top_indices)))

        self.layers[name] = self.layers[name].put_(top_indices, torch.zeros(len(top_indices)))
        self.layers[name] += small_values_tensor * momentum
        return top_indices, top_values

    def accumulate_gradients(self):
        """Accumulation gradients per iterations
        :return
        top_gradients : list [tensor], selected gradient values from all layers
        top_indices : list [tensor], indices of top_gradient values at the original gradient
        amounts_per_layer : list [int], numbers of elements selected from each layer"""
        gradient_tensors = []
        gradient_indices = []
        gradient_amounts = []
        layer_idx = 0

        for i, (name, p) in enumerate(self.main_model.named_parameters()):
            if i in self.full_update_layers:
                continue
            if len(p.grad.data.cpu().shape) == 4:
                top_indices, top_values = self.select_top_values_and_indices(p.grad.data.cpu(),
                                                                                           name, self.momentum)

                gradient_tensors.extend(top_values)
                gradient_indices.extend(top_indices)
                gradient_amounts.extend(np.ones(len(top_values)) * layer_idx)
                layer_idx += 1
        top_gradients = torch.FloatTensor(gradient_tensors)[None, None, None, ...]
        top_indices = torch.LongTensor(gradient_indices)
        amounts_per_layer = torch.LongTensor(gradient_amounts)

        return top_gradients, top_indices, amounts_per_layer

    def update_gradients(self, value):
        """update the final constructed sparse gradient before optimization
        :parameter
        value : list [tensor], constructed sparse gradients"""
        depth_idx = 0
        for i, (name, param) in enumerate(self.main_model.named_parameters()):
            if i in self.full_update_layers:
                continue
            if param.requires_grad:
                if len(param.grad.data.shape) == 4:
                    param.grad.data = value[depth_idx]
                    depth_idx += 1

    def avarage_gradients_sparse(self, value):
        """Avaraging sparse gradients obtained from the all nodes (synchronized)
        :parameter
        value : list [tensor], constracted sparse gradients of separate nodes
        :return
        avg_grads : tensor, avaraged over all nodes gradient tensor
         """
        idx = 0
        avg_grads = []
        for i, (name, param) in enumerate(self.main_model.named_parameters()):
            if i in self.full_update_layers:
                continue
            if len(param.grad.data.cpu().shape) == 4:
                layer = value[idx]
                idx += 1
                if self.rank == 0:
                    g_list = []
                    for i in range(self.size):
                        g_list.append(torch.zeros(layer.shape).to('cpu'))
                    dist.gather(tensor=layer.to('cpu'), dst=0, gather_list=g_list)
                    div = torch.zeros_like(g_list[0])
                    for i in range(len(g_list)):
                        div += (g_list[i] != 0).float()

                    div = torch.clamp(div, 1., len(g_list))
                    updated_grad = torch.sum(torch.stack(g_list), dim=0) / div

                    avg_grads.append(updated_grad.cuda(self.device_id))
                else:
                    dist.gather(tensor=layer.to('cpu'), dst=0, gather_list=[])

        return avg_grads

    def construct_grads(self, grads, indices,amounts):
        """Constructions from the separate indices sparse gradient tensor,
         with the same shape as original gradient, with top values at choosen indices,
         and zeros elsewhere
         :parameter
         grads : list [tensor], selected top values from each layer
         indices : list [tensor], indices of selected values
         amounts : list [int], numbers of elements selected from each layer
         :return
         new_grads : list [tensor], constructed sparse gradients
         """
        grads = grads[0, 0, 0]
        new_grads = []
        indices = indices.cuda(self.device_id)
        indices_amounts = amounts.cuda(self.device_id)
        conv_layer_idx = 0
        for i, (name, p) in enumerate(self.main_model.named_parameters()):
            if i in self.full_update_layers:
                continue
            tensor = p.grad.data.cpu()
            if len(tensor.shape) == 4:
                idc = indices[indices_amounts == conv_layer_idx]
                layer_grad = grads[indices_amounts == conv_layer_idx]

                updated_grad = torch.zeros_like(p.grad.data)
                updated_grad = updated_grad.put_(idc, layer_grad)
                new_grads.append(updated_grad)
                conv_layer_idx += 1
        return new_grads

    def transfer_gradients(self, grad_update_conv):
        """transfering avaraged sparse gradient to the all nodes for the optimization of the model at each node
        :parameter
        grad_update_conv : tensor, final avaraged sparse gradient tensor
        :return
        upd_grads : final gradient accessable at each node
        """
        upd_grads = []

        for idx in range(len(self.shapes)):
            updated = torch.zeros(self.shapes[idx])

            if self.rank == 0:
                reciever_list = []
                for i in range(self.size):
                    reciever_list.append(grad_update_conv[idx].to('cpu'))

                dist.scatter(tensor=updated, src=0, scatter_list=reciever_list)
            else:
                dist.scatter(tensor=updated, src=0, scatter_list=[])
            upd_grads.append(updated.cuda(self.device_id))

        return upd_grads


    def gradient_update(self, it):
        """main function for the gradient sparsification
        :parameter
        it : iteration of training
        """
        if it in self.iterations:
            self.current_persentage = self.percentages[self.iterations.index(it)]

        gradient_vector, gradient_indices, gradient_amounts = self.accumulate_gradients()

        updated_grads = self.construct_grads(gradient_vector.cuda(self.device_id), indices=gradient_indices, amounts=gradient_amounts)
        grad_update_conv = self.avarage_gradients_sparse(value=updated_grads)
        updated = self.transfer_gradients(grad_update_conv)
        self.update_gradients(updated)
        self.avarage_gradient_dense()
















