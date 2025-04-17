"""VertexJoinSelector Class."""
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


import numpy as np
import torch
from torch import nn

from .utils import to_tensor


# TODO: IMPLEMENT THIS PROPERLY
class VertexJointSelector(nn.Module):
    """Placeholder module for selecting joints from vertices.

    Used currently in HSMAL and SMAL models.

    """

    def __init__(self, **kwargs):
        """Initialise the JointVertexSelector.

        Sets the extra_joints_idxs to an empty array.
        """
        super().__init__()
        _ = kwargs  # to avoid unused argument warning
        extra_joints_idxs = np.array([])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        """Select the joints from the vertices.

        Args:
            vertices: The vertices of the model.
            joints: The joints of the model.

        Returns:
            joints: The selected joints from the vertices.

        NOTE: For VAREN, all joints are currently set under extra_joints..
        TODO: Fix this.

        """
        # The '.to(torch.long)'.
        # added to make the trace work in c++,
        # otherwise you get a runtime error in c++:
        # 'index_select(): Expected dtype int32 or int64 for index'
        extra_joints = torch.index_select(
            vertices,
            1,
            self.extra_joints_idxs.to(torch.long)
            )
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints
