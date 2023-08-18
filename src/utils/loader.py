import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from importlib import import_module
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset, WeightedRandomSampler, RandomSampler, SequentialSampler
from utils.utils import dict_update
import itertools
import yaml


def get_save_path(output_dir):
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info(f'=> Will save everything to {save_path}')
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.

   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

   """
   base_seed = torch.IntTensor(1).random_().item()
   np.random.seed(base_seed + worker_id)

def load_data_class(set_name):
    return getattr(import_module('datasets'), set_name)

def custom_collate_fn(batch):
    out_dict = {}
    keys = ('box_labels', 'dont_care', 'points', 'warped_points', 'shapes')
    for key, out_list in zip(keys, [[] for _ in range(len(keys))]):
        lb = False
        try:
            for i, sample in enumerate(batch):
                out = sample.pop(key)
                if lb := key in {'box_labels', 'dont_care'}:
                    out[:, 0] = i
                out_list.append(out)
            if lb:
                out_list = torch.cat(out_list, 0)
            out_dict.update({key: out_list})
        except KeyError:
            continue
    out_dict.update(default_collate(batch))

    return out_dict

def get_weights(datasets, set_weights):
    # normalize weights to dataset lengths
    lengths = [len(d) for d in datasets]
    longest = max(lengths)
    length_weights = [longest / l for l in lengths]
    # print(length_weights)

    weights_long = np.array(
        list(itertools.chain(*[[sw * lw] * l for sw, l, lw in zip(set_weights, lengths, length_weights)])))
    weights_long_norm = weights_long / weights_long.sum()
    # print("weights:", wei)
    return weights_long_norm

def dataLoader(config, action, DEBUG=False, return_points=False, export=False):
    """
    :param action: 'train' or 'val'
    :param DEBUG: no shuffle and load reduced dataset
    :param return_points: also return tensor of keypoint coordinates
    """

    data_transform = transforms.Compose([transforms.ToTensor()])
    if 'sub_configs' in config.keys():
        sub_configs = []
        for sub_config_path in config['sub_configs']:
            with open(sub_config_path, 'r') as f:
                sub_config = yaml.safe_load(f)
                sub_config = dict_update(sub_config, config)
                sub_configs.append(sub_config)
        config.update({'sub_configs': sub_configs})
    else:
        sub_configs = [config]

    datasets = []
    for sub_config in sub_configs:
        dataset_name = sub_config['data']['dataset']
        Dataset = load_data_class(dataset_name)
        logging.info(f"dataset: {dataset_name}")
        dataset_instance = Dataset(
            transform=data_transform,
            action=action,
            DEBUG=DEBUG,
            return_points=return_points,
            export=export,
            **sub_config['data']
        )
        datasets.append(dataset_instance)

    drop_last = not export and action != 'val'
    training_params = config.get('training_params', {})
    workers = training_params.get(f'workers_{action}', 8)
    logging.info(f"workers_{action}: {workers}")

    dataset = ConcatDataset(datasets)

    if action == 'val' or DEBUG or export:
        sampler = SequentialSampler(dataset)
    elif sampler_params := config.get('weighted_random_sampler'):
        weights = sampler_params.pop('weights')
        sampler_weights = get_weights(datasets, weights)
        sampler = WeightedRandomSampler(weights=sampler_weights, **sampler_params)
    else:   # action == 'train' and no concatdataset
        sampler = RandomSampler(dataset)
        # sampler = SequentialSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config['training_params'][f'{action}_batch_size'],
        sampler=sampler,
        pin_memory=True,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        collate_fn=custom_collate_fn
    )

    return loader

def dataLoader_test(config, dataset, export_task='train', DEBUG=False):
    # TODO: merge with dataLoader
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 8)
    batch_size = config.get('test_batch_size', 1)
    logging.info(f"workers_test: {workers_test}")

    if dataset == 'hpatches':
        raise NotImplementedError("hpatches has not yet been implemented")
        # from datasets.patches_dataset import PatchesDataset
        # if config['data']['preprocessing']['resize']:
        #     size = config['data']['preprocessing']['resize']
        # test_set = PatchesDataset(
        #     transform=data_transforms['test'],
        #     **config['data'],
        # )
        # test_loader = DataLoader(
        #     test_set, batch_size=1, shuffle=False,
        #     pin_memory=True,
        #     num_workers=workers_test,
        #     worker_init_fn=worker_init_fn
        # )
    # elif dataset == 'Coco' or 'Kitti' or 'Tum':
    else:
        logging.info(f"load dataset from : {dataset}")
        Dataset = load_data_class(dataset)
        test_set = Dataset(
            export=True,
            action=export_task,
            DEBUG=DEBUG,
            **config#['data'],
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn
        )
    return {'test_set': test_set, 'test_loader': test_loader}
