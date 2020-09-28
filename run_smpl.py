"""
Requires to run generate-labels-npy-multiview-smpl.py to get human36m-multiview-smpl-labels-GTbboxes.npy

run example:
python run_smpl.py --config experiments/human36m/train/human36m_vol_softmax_smpl.yaml
"""

import os
from os import path
import argparse
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import utils as dataset_utils
from mvn.datasets.human36m_smpl import Human36MMultiViewDataset
from train import setup_experiment, init_distributed

#from smpl.smpl_spin import SMPL
#from smplx import SMPL
from smpl.smpl_torch import SMPLModel as SMPL

def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = Human36MMultiViewDataset(
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
            shuffle=False, # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_smpl_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=0, #config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = Human36MMultiViewDataset(
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
        shuffle=False,
        collate_fn=dataset_utils.make_smpl_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=0, # config.dataset.val.num_workers,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/human36m/train/human36m_vol_softmax.yaml')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_dataset', type=str, default='val')

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logdir', type=str, default='logs')

    parser.add_argument('--device', type=int, default=0)

    #parser.add_argument('--smpl_dir', type=str, default='data/smplx/models')
    parser.add_argument('--smpl_dir', type=str, default='data/smpl/models')

    args = parser.parse_args()
    return args

# 3d keypoints: [batch_size, n_joints, 3]
def convert_h36m_to_smpl(keypoints_3d):
    batch_size = keypoints_3d.shape[0]
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    S24 = np.zeros((batch_size, 24,4))
    S24[:, global_idx, :3] = keypoints_3d
    S24[:, global_idx, 3] = 1.0

    return S24

# smpl_keypoints: torch [batch_size, 24, 4]
def convert_coord_to_angular(smpl_keypoints):
    batch_size = smpl_keypoints.shape[0]
    parent_joints = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                   8, 9, 12, 12, 12, 13, 14, 16, 17, 18,
                   19, 20, 21] # except base joint

    parent_bones = [-1, -1, -1, 1, 5, 3, 4, 5, 6, 7,
                    8, 9, 9, 9, 12, 13, 14, 16, 17, 18,
                    19, 20, 21]

    bone_vectors = smpl_keypoints[:,1:,:3] - smpl_keypoints[:,parent_joints, :3] # [batch_size, 23, 3]
    child_bone_vectors = bone_vectors.clone()

    angles = torch.zeros(*bone_vectors.shape) # when ignore base joint

    for child_bone_idx, parent_bone_idx in enumerate(parent_bones):
        # compute angle of parent bone
        parent_angle = torch.zeros(batch_size, 3)

        if parent_bone_idx != -1:
            parent_angle = [calc_angle_torch(bone_vectors[batch_i, parent_bone_idx].cpu()) for batch_i in range(batch_size)]
            parent_angle = torch.stack(parent_angle)

        # rotate child bone with the -angle of parent bone
        rotmat = batch_rodrigues(-parent_angle) # [batch_size, 3, 3]
        #child_bone_vectors[:,child_bone_idx,:] = torch.matmul(rotmat, bone_vectors[:,child_bone_idx,:])
        child_bone_vectors[:,child_bone_idx,:] = torch.einsum("ijk,ik->ij", (rotmat.cpu(), bone_vectors[:,child_bone_idx,:].cpu()))

        # calculate axis-angle rotation for the child bone
        child_angle = [calc_angle_torch(child_bone_vectors[batch_i, child_bone_idx].cpu()) for batch_i in range(batch_size)]
        child_angle = torch.stack(child_angle)

        angles[:,child_bone_idx] = child_angle

    # include base joint
    angles = torch.cat((angles, torch.zeros(batch_size, 1, 3)), 1)

    return angles

def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

# numpy way
def calc_angle(v):
    axis_vectors = np.eye(3, dtype=float)
    thetas = np.zeros(3, dtype=float) # x,y,z rotations

    for i in range(3):
        thetas[i] = np.arccos(np.dot(axis_vectors[i], v) / (norm(axis_vectors[i]) * norm(norm(v))))

    #thetas = 180 * thetas / np.pi

    return thetas

# pytorch way
# v : [3]. no batch support
def calc_angle_torch(v):
    axis_vectors = torch.eye(3)
    thetas = torch.zeros(3) # x,y,z rotations

    for i in range(3):
        thetas[i] = torch.acos(torch.dot(axis_vectors[i], v) / (torch.norm(axis_vectors[i]) * torch.norm(v)))

    #thetas = 180 * thetas / np.pi

    return thetas


def render_mesh(args, config, smpl_male, smpl_female, dataloader, device, is_train=True, experiment_dir=None):
    name = "train" if is_train else "val"
    model_type = config.model.name

    male_subjects = ['S6', 'S8', 'S9', 'S11']
    female_subjects = ['S1', 'S5', 'S7']

    iterator = enumerate(dataloader)
    for iter_i, batch in iterator:
        if batch is None:
            print("Found None batch")
            continue

        images_batch, keypoints_3d_gt, smpl_keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_smpl_batch(batch, device, config)

        batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])

        keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)

        scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

        # convert coordkeypoints into angular
        angles_3d_gt_smpl = convert_coord_to_angular(smpl_keypoints_3d_gt) # [batch_size, 23, 3]

        if n_views == 1:
            if config.kind == "human36m":
                base_joint = 6
            elif config.kind == "coco":
                base_joint = 11

            keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
            keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
            keypoints_3d_gt = keypoints_3d_gt_transformed


        # get betas
        beta_size = 10
        batch_size = 1
        dummy_betas = torch.from_numpy((np.random.rand(batch_size, beta_size)-0.5)*0.05)

        dummy_betas = dummy_betas.to(device)
        angles_3d_gt_smpl = angles_3d_gt_smpl.to(device)
        keypoints_3d_gt = keypoints_3d_gt.to(device)



        for batch_i in range(batch_size):
            # chose which model to use
            model = smpl_male if batch['subject'][batch_i] in male_subjects else smpl_female

            # run smpl
            """
            smpl_output = model(betas=dummy_betas[batch_i].unsqueeze(0).type(torch.float32), # [B x 10]
                                body_pose=angles_3d_gt_smpl[batch_i].reshape(1, -1), # [B x (joints x 3)]
                                transl=keypoints_3d_gt[batch_i, 0, :3].unsqueeze(0)) # [B x 3]

            """
            smpl_vertices = model(dummy_betas[batch_i].type(torch.float64),
                                angles_3d_gt_smpl[batch_i].type(torch.float64),
                                keypoints_3d_gt[batch_i, 0, :3].type(torch.float64))

            fname = f'{batch["subject"][batch_i]}_{batch["action"][batch_i]}_{batch["indexes"][batch_i]}'
            model.write_obj(smpl_vertices, fname+'.obj')

            import ipdb; ipdb.set_trace()
            print(f'iter_i:{iter_i}/{len(dataloader)}, batch_i:{batch_i}/{batch_size}')


def main():
    args = parse_args()
    print(args)

    # NO distributed settings
    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ['RANK']:
        master = int(os.environ['RANK']) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)


    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    """
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])
    """

    # load SMPL model
    # num_body_joints == 23
    # shap_space_dim == 300
    #smpl_male = SMPL(model_path=path.join(args.smpl_dir, 'SMPLX_MALE.pkl'), batch_size=1, gender='male', vertex_ids=None) # config.opt.batch_size
    #smpl_female = SMPL(model_path=path.join(args.smpl_dir, 'SMPLX_FEMALE.pkl'), batch_size=1, gender='female', vertex_ids=None)
    smpl_male = SMPL(model_path=path.join(args.smpl_dir, 'model_m.pkl'), device=device)
    smpl_female = SMPL(model_path=path.join(args.smpl_dir, 'model_f.pkl'), device=device)

    smpl_male.to(device)
    smpl_female.to(device)
    
    render_mesh(args, config, smpl_male, smpl_female, train_dataloader, device, True, experiment_dir)
    render_mesh(args, config, smpl_male, smpl_female, val_dataloader, device, False, experiment_dir)

if __name__ == '__main__':
    main()
