import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataset import TaobaoTrainSet
from sampler import RandomIdentitySampler, GroupedBatchSampler

def collate_fn(batch):

    imgs, pids, c = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    c = torch.tensor(c, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, c

def make_dataloader(json_list, batch_size, scale, mode, num_workers=0,net_type='effnet',crop=True):
        
    dataset = TaobaoTrainSet(json_list, scale, mode,net_type,crop)
    aspect_ids = [d['aspect_group'] for d in dataset.json_list]

    if mode == 'train':
        sampler = RandomSampler(dataset)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    elif mode == 'valid':
        sampler = SequentialSampler(dataset)

    gb_sampler = GroupedBatchSampler(
        sampler=sampler, 
        group_ids=aspect_ids,
        batch_size=batch_size,
        drop_uneven=(mode == 'train'))
    # (mode == 'train')
    return DataLoader(dataset, batch_sampler=gb_sampler,num_workers=num_workers,collate_fn=collate_fn,pin_memory=True)


def make_dataloader_without_crop(json_list, batch_size, scale, mode, num_workers=0, net_type='effnet', crop=True):
    dataset = TaobaoTrainSet(json_list, scale, mode, net_type, crop)
    type_ids = [int(d['type']=='image') for d in dataset.json_list]
    if mode == 'train':
        # sampler = RandomSampler(dataset)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    elif mode == 'valid':
        sampler = SequentialSampler(dataset)

    gb_sampler = GroupedBatchSampler(
        sampler=sampler,
        group_ids=type_ids,
        batch_size=batch_size,
        drop_uneven=(mode == 'train'))
    return DataLoader(dataset, batch_sampler=gb_sampler, num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)