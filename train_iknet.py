# This scrip assumes that keypoints_3d_gt follows smplx format
# You need to have human36m-multiview-smpl-labels-GTbboxes.npy for correct running
import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.iknet import IKNet_Baseline
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
#from mvn.datasets import human36m
from mvn.datasets import human36m_smpl as human36m
from mvn.datasets import utils as dataset_utils

from tqdm import tqdm

from smpl_libs import config as smpl_config
from mvn.utils.transforms3d import quat2mat

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='experiments/human36m/train/human36m_vol_softmax_smpl.yaml', help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")

    # iknet params
    parser.add_argument("--iknet_depth", type=int, default=6, help="number of layers in iknet model")
    parser.add_argument("--iknet_width", type=int, default=1024, help="number of hidden units in each layer")

    args = parser.parse_args()
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch(model, smpl, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None, best_metric=999.9):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(tqdm(dataloader, desc=f'{name}, epoch:{epoch}'))
        """
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)
        """

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print("Found None batch")
                    continue

                images_batch, keypoints_3d_gt, smpl_keypoints_3d_gt, keypoints_3d_validity_gt, smpl_keypoints_validity, proj_matricies_batch = dataset_utils.prepare_smpl_batch(batch, device, config)

                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])

                pred_quaternion = model(smpl_keypoints_3d_gt)

                # convert from quaternions into rotation matrix
                pred_rotmats = torch.stack([quat2mat(quat) for quat in pred_quaternion])

                # dummy betas
                beta_size = 10
                dummy_betas = torch.from_numpy((np.random.rand(batch_size, beta_size)-0.5)*0.05)

                # run smpl
                smpl_output = smpl(betas=dummy_betas, body_pose=pred_rotmats[:,1:], global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False)
                keypoints_3d_pred = smpl_output.joints

                ipdb.set_trace()

                n_joints = keypoints_3d_pred.shape[1]

                keypoints_3d_binary_validity_gt = (smpl_keypoints_validity > 0.0).type(torch.float32)

                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

                # calculate loss
                total_loss = 0.0
                loss = criterion(keypoints_3d_pred * scale_keypoints_3d, smpl_keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                metric_dict['total_loss'].append(total_loss.item())
                print('epoch: {}, {}: {}, volumetric_ce_loss: {}, total_loss: {}'.format(epoch, config.opt.criterion, metric_dict[f'{config.opt.criterion}'][-1], metric_dict['volumetric_ce_loss'][-1], total_loss.item()))

                if is_train:
                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step()

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])

                # plot visualization
                if master:
                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

                    n_iters_total += 1

    # calculate evaluation metrics
    comparative_metric = 999.9
    if master:
        if not is_train:
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric) # mean per pose relative error in human36m
            print('epoch: {}, dataset_metric: {}'.format(epoch, scalar_metric))

            # save the best results
            comparative_metric = scalar_metric if scalar_metric != 0.0 else comparative_metric
            if comparative_metric < best_metric:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                # dump results
                with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                    pickle.dump(results, fout)

                # dump full metric
                with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                    json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total, comparative_metric


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    model = IKNet_Baseline(args, config, device=device).to(device)

    # load smpl model
    smpl = SMPL(smpl_config.SMPL_MODEL_DIR,
                batch_size = config.opt.batch_size,
                create_transl=False).to(device)
    smpl.eval() # do not train smpl model

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval:
        opt = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.opt.lr
            weight_decay=1.0e-5
        )


    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    if not args.eval:
        print('training process')
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        best_metric = 999.9
        best_epoch = 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            n_iters_total_train, _ = one_epoch(model, smpl, criterion, opt, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer, best_metric=best_metric)
            n_iters_total_val, scalar_metric = one_epoch(model, smpl, criterion, opt, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, best_metric=best_metric)

            if master and scalar_metric < best_metric:
                best_metric = scalar_metric
                best_epoch = epoch
                # remove previous checkpoints
                checkpoints = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
                for checkpoint in checkpoints:
                    if checkpoint != '{:04}'.format(epoch):
                        shutil.rmtree(os.path.join(experiment_dir, "checkpoints", checkpoint))

                # save new checkpoint
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
                print(f"Best case saved at {epoch}th epoch. scalar_metric:{scalar_metric}")

            print(f"{n_iters_total_train} iters done.")
    else:
        print('evaluation process')
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)