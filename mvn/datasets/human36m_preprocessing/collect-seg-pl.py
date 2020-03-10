"""
    Read Segments_mat_gt_pl *.mat files from Human3.6M and convert them to a single *.npy file.
    Example of an original segmentation part label file:
    <path-to-Human3.6M-root>/S1/Segments_mat_gt_pl/Directions 1.54138969.mat

    Usage:
    python3 collect-seg-pl.py <path-to-Human3.6M-root> <num-processes>
"""
import os, sys
import numpy as np
import h5py

import pickle
from tqdm import tqdm

dataset_root = sys.argv[1]
data_path = os.path.join(dataset_root, "extracted")
subjects = [x for x in os.listdir(data_path) if x.startswith('S')]
assert len(subjects) == 7

destination_dir = os.path.join(dataset_root, "extra")
os.makedirs(destination_dir, exist_ok=True)
destination_file_path = os.path.join(destination_dir, "seg_pl-Human36M-GT.npy")

# Some bbox files do not exist, can be misaligned, damaged etc.
# dict is not fully verified. There could be potential error
from action_to_seg_pl_filename import action_to_seg_pl_filename
from collections import defaultdict

import cv2

nesteddict = lambda: defaultdict(nesteddict)

seg_pls_retval = nesteddict()

def load_seg_pls(data_path, subject, action, camera):
    print(subject, action, camera)

    def mask_to_seg_pl(mask):
        h_mask = mask.max(0)
        w_mask = mask.max(1)

        top = h_mask.argmax()
        bottom = len(h_mask) - h_mask[::-1].argmax()

        left = w_mask.argmax()
        right = len(w_mask) - w_mask[::-1].argmax()

        return top, left, bottom, right

    def resize_mask(mask):
        return cv2.resize(mask, (384, 384), interpolation=cv2.INTER_AREA)

    try:
        try:
            corrected_action = action_to_seg_pl_filename[subject][action]
        except KeyError:
            corrected_action = action.replace('-', ' ')

        # TODO use pathlib
        seg_pls_path = os.path.join(
            data_path,
            subject,
            'Segments_mat_gt_pl',
            '%s.%s.mat' % (corrected_action, camera))

        with h5py.File(seg_pls_path, 'r') as h5file:
            retval = np.empty((len(h5file['Feat']), 384, 384), dtype=np.int32)

            for frame_idx, mask_reference in enumerate(h5file['Feat'][:,0]):
                seg_pl_mask = np.array(h5file[mask_reference])
                retval[frame_idx] = resize_mask(seg_pl_mask) #mask_to_seg_pl(seg_pl_mask)
                
                """ We don't have to calculate bbox in processing segmentation masks
                top, left, bottom, right = retval[frame_idx]
                if right-left < 2 or bottom-top < 2:
                    print('right-left:{}, bottom-top:{}'.format(right-left, bottom-top))
                    print('right: {}, left: {}, bottom: {}, top: {}'.format(right, left, bottom, top))
                    raise Exception(str(seg_pls_path) + ' $ ' + str(frame_idx))
                """
    except Exception as ex:
        # reraise with path information
        raise Exception(str(ex) + '; %s %s %s' % (subject, action, camera))
    
    return retval, subject, action, camera

# retval['S1']['Talking-1']['54534623'].shape = (n_frames, 4) # top, left, bottom, right
def add_result_to_retval(args):
    seg_pls, subject, action, camera = args
    seg_pls_retval[subject][action][camera] = seg_pls

"""
import multiprocessing
num_processes = int(sys.argv[2])
pool = multiprocessing.Pool(num_processes)
async_errors = []

for subject in subjects:
    subject_path = os.path.join(dataset_root, 'processed', subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat') # folder with seg_pl *.mat files
    except ValueError:
        pass

    for action in actions:
        cameras = '54138969', '55011271', '58860488', '60457274'

        for camera in cameras:
            async_result = pool.apply_async(
                load_seg_pls,
                args=(data_path, subject, action, camera),
                callback=add_result_to_retval)
            async_errors.append(async_result)

pool.close()
pool.join()

# raise any exceptions from pool's processes
for async_result in async_errors:
    async_result.get()
"""

# refined multiprocessing
# m.p. for each camera to reduce memory size of each chunk
import multiprocessing
num_proc = int(sys.argv[2])
pool = multiprocessing.Pool(num_proc)
async_errors = []

pbar = tqdm(total=len(subjects))
for subject in subjects:
    subject_path = os.path.join(dataset_root, 'processed', subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat')
    except ValueError:
        pass

    for action in actions:
        cameras = '54138969', '55011271', '58860488', '60457274'

        async_res = [pool.apply_async(
                    load_seg_pls,
                    args=(data_path, subject, action, camera),
            callback=add_result_to_retval) for camera in cameras]

        for r in async_res:
            async_errors.append(r)
            r = r.get()
            add_result_to_retval(r)
        
    pbar.update(1)
pbar.close()
pool.close()
pool.join()

# raise any exceptions from pool's processes
for async_result in async_errors:
    async_result.get()
"""
# single processing to prevent memory issue in multiprocessing
# serializing more than 4GiB results in error
pbar = tqdm(total=len(subjects))
for subject in subjects:
    subject_path = os.path.join(dataset_root, 'processed', subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat')
    except ValueError:
        pass

    for action in actions:
        cameras = '54138969', '55011271', '58860488', '60457274'

        for camera in cameras:
            res = load_seg_pls(data_path, subject, action, camera)
            add_result_to_retval(res)
    pbar.update(1)
pbar.close()
"""

def freeze_defaultdict(x):
    x.default_factory = None
    for value in x.values():
        if type(value) is defaultdict:
            freeze_defaultdict(value)

# convert to normal dict
freeze_defaultdict(seg_pls_retval)
#np.save(destination_file_path, seg_pls_retval)
dump_file = open(destination_file_path, 'wb')
pickle.dump(seg_pls_retval, dump_file)
dump_file.close()


