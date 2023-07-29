from functools import partial
from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader
import torch
from .sampler import GroupSampler, DistributedGroupSampler, DistributedSampler, FastRandomIdentitySampler, ClassAwareSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def collate_fn_XChestFuse(batch):
    for item in batch:
        print(item["img"].data.shape)
        # exit()
    # print(f"type of batch: {type(batch)}")
    # exit()
    
    # images = torch.stack(images, dim=0)
   
    img = [item['img'].data for item in batch]
    img_meta = [item['img_meta'].data for item in batch]
    gt_labels = [item['gt_labels'].data for item in batch]
    
    img = torch.cat(img, dim=0) # Ghép nối các tensor ảnh thành một tensor duy nhất
    img_meta = torch.cat(img_meta, dim=0)   
    gt_labels = torch.cat(gt_labels, dim=0)   # Ghép nối các tensor nhãn thành một tensor duy nhất
    
    return {
        'img': img,
        'img_meta': img_meta,
        'gt_labels': gt_labels,
    }



def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     sampler='Group',
                     sampler_cfg = None,
                     custom_collate = "",
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
            # sampler = DistributedSampler(
            #     dataset, world_size, rank, shuffle=True)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:

        if 'FaseRandomIdentity' in sampler:
            assert sampler_cfg is not None
            sampler = FastRandomIdentitySampler(dataset,
                                                sampler_cfg.num_classes,
                                                sampler_cfg.num_instances,
                                                sampler_cfg.select_classes,
                                                sampler_cfg.select_instances,
                                                imgs_per_gpu)
        elif 'ClassAware' in sampler:
            if sampler_cfg is not None:
                reduce = sampler_cfg.get('reduce', 4)
            else:
                reduce = 4
            sampler = ClassAwareSampler(data_source=dataset, reduce=reduce)
        elif 'Group' in sampler:
            sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        # elif "DistributedSampler" in sampler:

        else:
            raise NameError
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu
    
    if custom_collate is not None:
        print("custom_collate is not None")
        if "XChestFuse" in custom_collate:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn_XChestFuse,
                pin_memory=False,
                drop_last=True,
                **kwargs)
            return data_loader


    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        drop_last=True,
        **kwargs)
    print("data_loader end\n")
    return data_loader
