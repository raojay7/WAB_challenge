import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataset import MetricInferSet
from sampler import RandomIdentitySampler, GroupedBatchSampler

def collate_fn(batch):
    imgs, idx = zip(*batch)
    return torch.stack(imgs, dim=0), idx

def make_dataloader(instance_list, batch_size, scale,net_type='effnet',crop=True):
        
    dataset = MetricInferSet(instance_list, scale , net_type,crop)
    aspect_ids = [d['aspect_group'] for d in dataset.instance_list]

    sampler = SequentialSampler(dataset)

    gb_sampler = GroupedBatchSampler(
        sampler=sampler, 
        group_ids=aspect_ids,
        batch_size=batch_size,
        drop_uneven=False)

    return DataLoader(dataset, batch_sampler=gb_sampler, collate_fn=collate_fn)


def make_dataloader_without_crop(instance_list, batch_size, scale, net_type='effnet',crop=False):
    dataset = MetricInferSet(instance_list, scale, net_type,crop)
    type_ids = [int(d['type']=='image') for d in dataset.instance_list]

    sampler = SequentialSampler(dataset)

    gb_sampler = GroupedBatchSampler(
        sampler=sampler,
        group_ids=type_ids,
        batch_size=batch_size,
        drop_uneven=False)

    return DataLoader(dataset, batch_sampler=gb_sampler, collate_fn=collate_fn)