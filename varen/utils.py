"""Collection of utility functions and classes for the VARENA model."""
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

from dataclasses import dataclass, fields
from typing import NewType, Optional, Union

import numpy as np
import torch

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


@dataclass
class ModelOutput:
    """Dataclass to hand the varen Output.

    Attributes:
        vertices: The vertices of the SMPL model.
        joints: The joints of the SMPL model.
        full_pose: The full pose of the SMPL model.
        global_orient: The global orientation of the SMPL model.
        transl: The translation of the SMPL model.
        v_shaped: The shape of the vertices.

    Note:
        The attributes are optional and can be None.
        The attributes are not guaranteed to be present in the output.

    """

    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        """Return the value of the attribute with the given key."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Return the value of the attribute with the given key."""
        return getattr(self, key, default)

    def __iter__(self):
        """Return an iterator over the keys of the attributes."""
        return self.keys()

    def keys(self):
        """Return an iterator over the keys of the attributes."""
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        """Return an iterator over the values of the attributes."""
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        """Return an iterator over the keys and values of the attributes."""
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class MuscleDeformer:
    """Dataclass to handle the muscle deformer output.

    Attributes:
        betas_muscle: The muscle deformation parameters.
        Bm: The muscle deformation matrix.
        muscle_idxs: The indices of the muscles.

    """

    # betas_muscle, Bm, muscle_idxs. Not Optional:
    betas_muscle: Tensor
    Bm: Tensor
    muscle_idxs: Tensor


@dataclass
class SMALOutput(ModelOutput):
    """Dataclass to handle the SMAL output.

    Attributes:
        betas: The shape parameters of the SMAL model.
        body_pose: The pose parameters of the SMAL model.

    """

    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class VARENOutput(ModelOutput):
    """Dataclass to handle the VAREN output.

    Attributes:
        mdv: Muscle Deformation 3D Offsets.
        surface_keypoints: The surface keypoints of the VAREN model.
        body_pose: The pose parameters of the VAREN model.
        body_betas: The shape parameters of the VAREN model.
        muscle_betas: The muscle deformation parameters.
        muscle_activations: The muscle activations of the VAREN model.

    """

    mdv: Optional[Tensor] = None
    surface_keypoints: Optional[Tensor] = None,
    body_pose: Optional[Tensor] = None
    body_betas: Optional[Tensor] = None
    muscle_betas: Optional[Tensor] = None
    muscle_activations: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    """Find the kinematic chain of a joint.

    Args:
        joint_id: The id of the joint.
        kinematic_tree: The kinematic tree of the model.

    Returns:
        kin_chain: The kinematic chain of the joint.

    """
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    """Convert a numpy array to a torch tensor."""
    if torch.is_tensor(array):
        return array
    return torch.tensor(array, dtype=dtype)


class Struct:
    """A simple class to create a struct-like object.

    This class allows you to create an object with attributes
    that can be accessed like a struct.

    Example:
        obj = Struct(a=1, b=2)
        print(obj.a)  # 1
        print(obj.b)  # 2

    """

    def __init__(self, **kwargs):
        """Initialise the struct with the given keyword arguments."""
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    """Convert a tensor or space array to a numpy array."""
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def rot_mat_to_euler(rot_mats):
    """Convert rotation matrices to euler angles.

    Args:
        rot_mats: Rotation matrices of shape (..., 3, 3).

    Returns (torch.Tensor):
        euler angles of shape (..., 3).

    """
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Taken from pytorch3d.transforms

    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [
        torch.cos(half_angles),
        axis_angle * sin_half_angles_over_angles
        ],
        dim=-1
    )
    return quaternions
