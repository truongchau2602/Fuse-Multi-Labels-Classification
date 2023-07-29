import torch.nn as nn

from .base import BaseClassifier
from .. import builder
from ..registry import CLASSIFIERS
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
from ..backbones.backbone_swin import build_backbone
from ..necks.transformer import build_transformer
from ..backbones.position_encoding import build_position_encoding
import math
import argparse

@CLASSIFIERS.register_module
class Query2LabelClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 head = None,
                 neck=None,
                 keep_input_proj = None,
                 lock_back=False,
                 lock_neck=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 savefeat=False,
                 ):
        super(Query2LabelClassifier, self).__init__()

        self.args_backbone = self.create_args_from_dict(backbone, "backbone")
        self.args_neck = self.create_args_from_dict(neck, "neck")

        self.backbone = build_backbone(self.args_backbone)
        if neck is not None and neck["type"] == "Transformer":
            self.neck = build_transformer(self.args_neck)
        else:
            assert neck is not None, 'We must have a neck'
            
        self.model = Query2Label(
        backbone = self.backbone,
        transfomer = self.neck,
        num_class = self.args_neck.num_class
        )

        if not self.args_neck.keep_input_proj:
            model.input_proj = nn.Identity()
            print("set model.input_proj to Indentify!")

        self.head = builder.build_head(head)


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.lock_back=lock_back
        self.lock_neck=lock_neck
        self.savefeat=savefeat
        if self.savefeat and not self.with_neck:
            assert neck is not None, 'We must have a neck'
            assert train_cfg is None, 'this is only at testing stage'
        if self.lock_back:
            print('\033[1;35m >>> backbone locked !\033[0;0m')
        if self.lock_neck:
            print('\033[1;35m >>> neck locked !\033[0;0m')
        self.init_weights(pretrained=pretrained)
        self.count = CountMeter(num_classes=20)

        # freq = torch.tensor([5,6,6,5,7,16,12,5,9,6,5,5,10,21,6,5],dtype=torch.float)
        # self.cla_weight = torch.mean(torch.sqrt(freq))*(torch.ones(freq.shape,dtype=torch.float) / torch.sqrt(freq)).cuda()
        # print(self.cla_weight)

    def create_args_from_dict(self, cfg_dict, name):
        print(name)
        args = argparse.Namespace()
        for key, value in cfg_dict.items():
            print(f"{key}:{value}")
            setattr(args, key, value)
        return args

    def init_weights(self, pretrained=None):
        super(Query2LabelClassifier, self).init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        # if self.with_neck:
        #     if isinstance(self.neck, nn.Sequential):
        #         for m in self.neck:
        #             m.init_weights()
        #     else:
        #         self.neck.init_weights()
        # self.head.init_weights()
        return

    def extract_feat(self, img):
        # if self.lock_back:
        #     with torch.no_grad():
        #         x = self.backbone(img)
        # else:
        #     x = self.backbone(img)

        # if self.with_neck:
        #     if self.lock_neck:
        #         with torch.no_grad():
        #             x = self.neck(x)
        #     else:
        #         x = self.neck(x)
        x = self.model(img)

        return x

    def forward_train(self,
                      img,
                      img_metas, 
                      gt_labels):
        # if self.lock_back:
        #     with torch.no_grad():
        #         x = self.extract_feat(img)
        # else:
        #     x = self.extract_feat(img)
        x = self.extract_feat(img)
        outs = self.head(x)

        loss_inputs = (outs, gt_labels)
        losses = self.head.loss(*loss_inputs)

        # self.count.update(gt_labels)

        return losses

    # def simple_test(self, img, img_meta=None, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.head(x)
    #     return outs

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.head(x)

        if self.savefeat:
            return outs, x

        return outs

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

''' This part is only for resampling
'''
class CountMeter(object):
    def __init__(self, num_classes, vector_len=1, end_n=15500): # coco reduce 4: 22560 # voc reduce 1: 15500
        self.num_classes = num_classes
        self.end_n = end_n
        self.vector_len = vector_len
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes, dtype=torch.int64).cuda()
        self.all_features = None
        self.all_labels = None
        self.n = 0

    def update(self, gt_labels, features=None):
        n = gt_labels.shape[0]
        self.count += torch.sum(gt_labels, dim=0)
        if self.all_labels is None:
            self.all_labels = gt_labels.cpu().numpy()
            self.all_features = features.cpu().numpy()
        else:
            self.all_labels = np.concatenate((self.all_labels, gt_labels.cpu().numpy()))
            self.all_features = np.concatenate((self.all_features, features.cpu().numpy()))
        self.n += n
        if self.n >= self.end_n:
            self.save_and_exit()

    def save_and_exit(self):
        import mmcv
        data = dict(count=self.count.cpu().numpy(), all_labels=self.all_labels)
        mmcv.dump(data, './mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl')
        print('resample result saved at :{}'.format('./mllt/appendix/VOCdevkit/longtail2012/resample_results_b6.pkl'))
        exit()


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Query2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)


    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        out = self.fc(hs[-1])
        # import ipdb; ipdb.set_trace()
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    

    return model