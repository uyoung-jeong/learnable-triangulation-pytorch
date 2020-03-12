"""
    Read seg_pl *.mat files from Human3.6M and convert to *.jpg attention map files.
    One attention map image is generated for each image made from process_all.py in human36m-fetch.

    Usage:
    python collect-seg-pl.py <path-to-Human3.6M-root> <num-processes> 
"""

import os, sys
from os.path import join
import h5py
import PIL
import cv2
import numpy as np
from tqdm import tqdm

from action_to_seg_pl_filename import action_to_seg_pl_filename

data_root = sys.argv[1]
img_data_path = os.path.join(data_root, 'processed')
subjects = [x for x in os.listdir(img_data_path) if x.startswith('S')]
if len(subjects) < 1:
    print("There is no subject to process.")
    exit()

img_actions = [x for x in os.listdir(join(img_data_path, subjects[0])) if x.find('-')!=-1]
if len(img_actions) < 1:
    print("There is no action to process.")
    exit()

save_dir = join(data_root, "extra", "seg_pl")
# prepare dirs
print("Making directories in the destination path:")
print(save_dir)
os.makedirs(save_dir, exist_ok=True)
for subj in subjects:
    subj_path = join(save_dir, subj)
    os.makedirs(subj_path, exist_ok=True)

    # make dirs for each action
    for act in img_actions:
        act_path = join(subj_path, act)
        os.makedirs(act_path, exist_ok=True)


# multiprocessing routine
def routine(img_action_path, cam, subj, action, files, save_dir):
    img_cam_path = os.path.join(img_action_path, cam)
    img_files = []
    try:
        img_files = [f for f in os.listdir(img_cam_path) if os.path.isfile(os.path.join(img_cam_path, f))]
    except FileNotFoundError as e:
        if (subj == 'S11') and (action == 'Directions-2') and (cam == '54138969'):
            # print('this camera data is blacklisted by process_all.py')
            return
        print(e)
    indices = np.arange(0, len(img_files), 400)
    
    # select matching file
    selected_f = ''
    selected_a = action
    try:
        selected_a = action_to_seg_pl_filename[subj][action]
    except KeyError:
        selected_a = action.replace('-', ' ')

    for f in files:
        fname = f.split('/')[-1]
        tokens = fname.split('.')
        if (tokens[0] == selected_a) and (tokens[1] == cam) and (tokens[-1] == 'mat'):
            selected_f = f
            break
    
    if selected_f == '':
        print('matching seg map for {} not found'.format(action))
        return

    with h5py.File(selected_f, 'r') as mat:
        for img_f in img_files:
            # load image and save
            img_f_path = os.path.join(img_cam_path, img_f)
            img = cv2.imread(img_f_path)
            img_fname = img_f.replace('img_', '{}_{}_'.format(selected_a, cam))
            cv2.imwrite(join(save_dir, subj, img_fname), img)

            # get index of file
            img_index = img_f
            img_index = int(img_index.replace('.jpg', '').replace('img_', ''))-1
            
            # load mask
            mask_ref = None
            try:
                mask_ref = mat['Feat'][img_index, 0]
            except ValueError as e:
                print(e)
                print('action:{}, camera:{}, img_idx:{}'.format(action, cam, img_index))
                continue
            mask = np.array(mat[mask_ref], dtype=np.float32)

            # augment mask
            augmented = np.rot90(mask, 3) # rotate 90 deg. clockwise
            augmented = np.fliplr(augmented) # flip horizontally

            # add mask to the img
            alpha = simple_alpha(augmented)
            alpha = cv2.GaussianBlur(alpha, (5,5),0) # gaussian smoothing
            r,g,b = cv2.split(img)
            r = r*alpha
            g = g*alpha
            b = b*alpha
            rgba = cv2.merge((r,g,b))

            # save
            #cv2.imwrite(join(save_dir, subj, '{}_{}_{}_raw.jpg'.format(selected_a, cam, img_index)), mask)
            cv2.imwrite(join(save_dir, subj, '{}_{}_{}_aug.jpg'.format(selected_a, cam, img_index)), augmented)
            cv2.imwrite(join(save_dir, subj, '{}_{}_{}_alpha.jpg'.format(selected_a, cam, img_index)), rgba)

# simple function for making alpha channel
def simple_alpha(x):
    y = np.copy(x)
    nonzero_indices = np.nonzero(y)
    y[nonzero_indices] = 1
    return y

import multiprocessing as mp
n_proc = int(sys.argv[2])
max_proc = mp.cpu_count()
actual_proc = np.amin([max_proc, n_proc])
pool = mp.Pool(actual_proc)
print("Multiprocessing with {} threads".format(actual_proc))
async_errors = []

seg_base_path = os.path.join(data_root, 'extracted')
subdir = 'Segments_mat_gt_pl'

cameras = ['54138969', '55011271', '58860488', '60457274']

# iterate over subjects
for subj in subjects:
    subj_path = join(seg_base_path, subj, subdir) # seg subj path
    img_subj_path = join(img_data_path, subj) # img subj path

    # get .mat files in the specified subject
    files = []
    for r,d,f in os.walk(subj_path):
        for file in f:
            if '.mat' in file:
                files.append(os.path.join(r, file))

    actions = [o for o in os.listdir(img_subj_path) if (os.path.isdir(join(img_subj_path, o)) and o.find('MySegmentsMat') == -1)]
    
    pbar = tqdm(total=len(actions), desc=subj)
    for action in actions:
        img_action_path = join(img_subj_path, action, 'imageSequence-undistorted')
        # multiprocessing
        async_res = [pool.apply_async(
                        routine,
                        args=(img_action_path, cam, subj, action, files, save_dir)
                        )for cam in cameras]
        for res in async_res:
            res.get()
        pbar.update(1)
        """
        for cam in cameras:
            routine(img_action_path, cam, subj, action, files, save_dir) 
            pbar.update(1)
        """
    pbar.close()
pool.close()
pool.join()

print('Finished Execution')
