import numpy as np

import torch
import torch.nn as nn

# pytorch version of quaternions-to-rotation matrix
# original source: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
# no batch support
# q: 4 element array-like
def quat2mat(q):
    w,x,y,z = q
    Nq = w*w + x*x + y*y + z*z

    _FLOAT_EPS = torch.finfo(torch.float32).eps

    if Nq < _FLOAT_EPS:
        reuturn torch.eye(3)

    s = torch.divide(2.0, Nq)
    X = x*s
    Y = y*s
    Z = z*s

    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z

    return torch.tensor(
        [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
         [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
         [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
