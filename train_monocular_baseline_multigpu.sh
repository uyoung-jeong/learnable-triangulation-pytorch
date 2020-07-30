#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2345 train.py --config experiments/human36m/train/human36m_vol_softmax_monocular.yaml --logdir logs
