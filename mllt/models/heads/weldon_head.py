import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function

from mmcv.cnn import constant_init, kaiming_init

from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


class WeldonPool2dFunction(torch.autograd.Function):

    def __init__(self):
        super(WeldonPool2dFunction, self).__init__()
        # self.kmax = kmax
        # self.kmin = kmin
    @staticmethod
    def get_number_of_instances(k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)
    @staticmethod
    def forward(ctx, input,  kmax, kmin):
        # get batch information
        
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        # get number of regions
        n = h * w

        # get the number of max and min instances
        ctx.kmax = WeldonPool2dFunction.get_number_of_instances(kmax, n)
        ctx.kmin = WeldonPool2dFunction.get_number_of_instances(kmin, n)

        # sort scores
        sorted, indices = input.new(), input.new().long()
        torch.sort(input.view(batch_size, num_channels, n), dim=2, descending=True, out=(sorted, indices))

        # compute scores for max instances
        ctx.indices_max = indices.narrow(2, 0, ctx.kmax)
        ctx.output = sorted.narrow(2, 0, kmax).sum(2).div_(ctx.kmax)

        if kmin > 0:
            # compute scores for min instances
            ctx.indices_min = indices.narrow(2, n-kmin, kmin)
            ctx.output.add_(sorted.narrow(2, n-kmin, kmin).sum(2).div_(kmin)).div_(2)

        # save input for backward
        ctx.save_for_backward(input)
        # return output with right size
        return ctx.output.view(batch_size, num_channels)
    @staticmethod
    def backward(ctx, grad_output):

        # get the input
        input, = ctx.saved_tensors

        # get batch information
        batch_size = input.size(0)
        num_channels = input.size(1)
        h = input.size(2)
        w = input.size(3)

        # get number of regions
        n = h * w

        # get the number of max and min instances
        kmax = WeldonPool2dFunction.get_number_of_instances(ctx.kmax, n)
        kmin = WeldonPool2dFunction.get_number_of_instances(ctx.kmin, n)

        # compute gradient for max instances
        ctx.grad_output_max = grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, kmax)
        ctx.grad_input = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, ctx.indices_max, ctx.grad_output_max).div_(ctx.kmax)

        if kmin > 0:
            # compute gradient for min instances
            ctx.grad_output_min = ctx.grad_output.view(batch_size, num_channels, 1).expand(batch_size, num_channels, ctx.kmin)
            grad_input_min = grad_output.new().resize_(batch_size, num_channels, n).fill_(0).scatter_(2, ctx.indices_min, ctx.grad_output_min).div_(ctx.kmin)
            ctx.grad_input.add_(ctx.grad_input_min).div_(2)

        return ctx.grad_input.view(batch_size, num_channels, h, w)


class WeldonPool2d(nn.Module):

    def __init__(self, kmax=1, kmin=None):
        super(WeldonPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax

    def forward(self, input):
        # return WeldonPool2dFunction(self.kmax, self.kmin)(input)
        return WeldonPool2dFunction.apply(input, self.kmax, self.kmin)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ')'



@HEADS.register_module
class WeldonHead(nn.Module):
    """ Weldon classification head,
        https://ieeexplore.ieee.org/abstract/document/8242666
    """

    def __init__(self,
                 in_channels=2048,
                 num_classes=80,
                 kmax=1,
                 kmin=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0)):
        super(WeldonHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.transfer = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.spatial_pooling = WeldonPool2d(kmax, kmin)
       
        self.debug_imgs = None

    def init_weights(self):
        kaiming_init(self.transfer)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.transfer(x)
        x = self.spatial_pooling(x)
        cls_score = x.view(x.size(0), -1)
        return cls_score

    def loss(self,
             cls_score,
             labels,
             reduction_override=None):
        losses = dict()
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            None,
            avg_factor=None,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)
        return losses
