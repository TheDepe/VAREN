# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn

from .utils import to_tensor



class VertexJointSelector(nn.Module):

    def __init__(self, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = np.array([])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(torch.long)) #The '.to(torch.long)'.
                                                                                            # added to make the trace work in c++,
                                                                                            # otherwise you get a runtime error in c++:
                                                                                            # 'index_select(): Expected dtype int32 or int64 for index'
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints
    
