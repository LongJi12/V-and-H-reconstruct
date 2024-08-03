import torch
import numpy as np
import smplx
import os
from smplx import SMPL as _SMPL
from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints

from scipy import constants

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        BASE_DATA_DIR='/DATACENTER1/long.ji/BAAM/BAAM-main/data/base_data'
        JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(BASE_DATA_DIR, 'J_regressor_extra.npy')
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        # self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        # extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        # joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        # joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                            #  joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output