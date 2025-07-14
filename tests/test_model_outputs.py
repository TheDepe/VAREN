import pytest
import torch

from varen import VAREN
from varen.utils import VARENOutput


def test_forward_pass_full(varen_model):
    output = varen_model()
    assert isinstance(output, VARENOutput), "Output should be a VARENOutput instance."

    expected_keys = [
        "vertices",
        "joints",
        "mdv",
        "muscle_betas",
        "muscle_activations",
        "body_betas",
        "body_pose",
        "surface_keypoints",
        "full_pose",
        "global_orient"
    ]

    # Collect missing keys
    missing_keys = [key for key in expected_keys if key not in output]

    # Collect keys that should be None but aren't
    should_be_not_none_keys = ["mdv", "muscle_betas", "muscle_activations"]
    none_keys = [key for key in should_be_not_none_keys if output.get(key) is None]

    # Raise errors if any problems found
    errors = []
    if missing_keys:
        errors.append(f"Missing keys: {', '.join(missing_keys)}")
    if none_keys:
        errors.append(f"Expected None values for: {', '.join(none_keys)}")

    if errors:
        raise AssertionError(" | ".join(errors))

def test_forward_pass_no_muscles(varen_model_no_muscles):
    output = varen_model_no_muscles()
    assert isinstance(output, VARENOutput), "Output should be a VARENOutput instance."

    expected_keys = [
        "vertices",
        "joints",
        "mdv",
        "muscle_betas",
        "muscle_activations",
        "body_betas",
        "body_pose",
        "surface_keypoints",
        "full_pose",
        "global_orient"
    ]

    # Collect missing keys
    missing_keys = [key for key in expected_keys if key not in output]

    # Collect keys that should be None but aren't
    should_be_none_keys = ["mdv", "muscle_betas", "muscle_activations"]
    non_none_values = {
        key: output.get(key) for key in should_be_none_keys if output.get(key) is not None
    }

    # Raise errors if any problems found
    errors = []
    if missing_keys:
        errors.append(f"Missing keys: {', '.join(missing_keys)}")
    if non_none_values:
        for key, val in non_none_values.items():
            errors.append(f"Expected '{key}' to be None but got: {val}")

    if errors:
        raise AssertionError(" | ".join(errors))


def test_vertex_output(varen_model):
    output = varen_model()
    assert isinstance(output, VARENOutput), "Output should be a VARENOutput instance."
    vertices = output.vertices
    template = varen_model.v_template

    assert vertices is not None, "Vertices should not be None."
    assert isinstance(vertices, torch.Tensor), f"Vertices should be a torch.Tensor, got {type(vertices)}."
    assert vertices.dim() == 3, f"Vertices should have 3 dimensions. Got: {vertices.dim()}."
    assert vertices.shape[1:] == template.shape, f"Vertices shape should match template shape {template.shape}, got: {vertices.shape[1:]}."
    assert torch.isfinite(vertices).all(), "Vertices should not contain NaN or Inf values."

def test_joints_output(varen_model):
    output = varen_model()
    assert isinstance(output, VARENOutput), "Output should be a VARENOutput instance."
    joints = output.joints

    assert joints is not None, "Joints should not be None."
    assert isinstance(joints, torch.Tensor), f"Joints should be a torch.Tensor, got {type(joints)}."
    assert joints.dim() == 3, f"Joints should have 3 dimensions. Got: {joints.dim()}."
    # NOTE: We include + 1 for the global joint at location 0. In the model
    # files, we have a double up on joints 0 and 1. Both 'spine' and 'pelvis'
    # are located at 0,0,0. Overall we have 37 UNIQUE joints, but 38 joints
    # overall... The body_pose vector does not rotate around the GO, so it is
    # 37 long. Once we add the global joint, we have 38 'joints'.

    # NOTE 2: To change this, we would either want to move one of the joints,
    # so that there is no double up, OR we remove one of them and adjust the
    # sizing of everything.
    assert joints.shape[1] == varen_model.NUM_JOINTS + 1, f"Joints shape should match template joints shape {varen_model.NUM_JOINTS + 1}, got: {joints.shape}."
    assert torch.isfinite(joints).all(), "Joints should not contain NaN or Inf values."