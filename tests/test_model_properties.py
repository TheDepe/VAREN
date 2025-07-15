"""Test model properties of VAREN."""
from pathlib import Path

import numpy as np
import torch

from varen import VAREN


def test_gradients(
        varen_model,
        pose_cpu,
        shape_cpu,
        global_orient_cpu,
        translation_cpu
        ):
    """Test to see if gradients are computed correctly."""
    def make_input(tensor):
        noise = torch.rand_like(
            tensor, dtype=tensor.dtype, device=tensor.device
            )
        return (tensor + noise).detach().requires_grad_()

    dummy_input = {
        'body_pose': make_input(pose_cpu),
        'betas': make_input(shape_cpu),
        'global_orient': make_input(global_orient_cpu),
        'transl': make_input(translation_cpu),
    }

    output = varen_model(**dummy_input)
    output.vertices.sum().backward()

    failed = []
    for key, tensor in dummy_input.items():
        if tensor.grad is None:
            failed.append(key)

    assert not failed, f"Gradients not computed for: {', '.join(failed)}"


def test_gradients_without_muscles(
        varen_model_no_muscles,
        pose_cpu,
        shape_cpu,
        global_orient_cpu,
        translation_cpu
        ):
    """Test to see if gradients are computed correctly."""
    def make_input(tensor):
        noise = torch.rand_like(
            tensor,
            dtype=tensor.dtype,
            device=tensor.device
        )
        return (tensor + noise).detach().requires_grad_()

    dummy_input = {
        'body_pose': make_input(pose_cpu),
        'betas': make_input(shape_cpu),
        'global_orient': make_input(global_orient_cpu),
        'transl': make_input(translation_cpu),
    }

    output = varen_model_no_muscles(**dummy_input)
    output.vertices.sum().backward()

    failed = []
    for key, tensor in dummy_input.items():
        if tensor.grad is None:
            failed.append(key)

    assert not failed, f"Gradients not computed for: {', '.join(failed)}"


def test_faces(varen_model):
    """Test to see if the faces are correctly set."""
    faces = varen_model.faces
    assert isinstance(faces, np.ndarray), "Faces should be a torch.Tensor"
    assert faces.ndim == 2, "Faces should be a 2D tensor"  # noqa: PLR2004
    assert faces.shape[1] == 3, "Faces should have 3 vertices per face"  # noqa: PLR2004
    assert faces.dtype == np.int64, (
        f"Faces should be of type int64, got {faces.dtype}"
    )
    assert faces.shape[0] > 0, "Faces tensor should not be empty"


def test_model_initialization_with_default_values(
        varen_model_dir,
        varen_model_file_name,
        ckpt_file
    ):
    """Return a VAREN model instance."""
    varen_pkl = Path(varen_model_file_name)
    path, ext = varen_pkl.stem, varen_pkl.suffix

    BATCH_SIZE = 2
    NUM_BETAS = 15

    varen_model = VAREN(
        model_path=varen_model_dir,
        model_file_name=path,
        ext=ext,
        use_muscle_deformations=True,
        ckpt_file=ckpt_file,
        num_betas=NUM_BETAS,  # not default
        batch_size=BATCH_SIZE,  # not default
    )

    assert varen_model.betas is not None, "Betas should not be None."
    assert varen_model.betas.shape == (BATCH_SIZE, NUM_BETAS), (
        "Betas shape should be (2, 15)."
    )
    assert varen_model.global_orient is not None, (
        "Global orientation should not be None."
    )
    assert varen_model.global_orient.shape == (BATCH_SIZE, 3), (
        "Global orientation shape should be (2, 3)."
    )
    assert varen_model.body_pose is not None, "Body pose should not be None."
    assert varen_model.body_pose.shape == (
        BATCH_SIZE, varen_model.NUM_JOINTS * 3
        ), (
            f"Body pose shape should be ({BATCH_SIZE},"
            f"{varen_model.NUM_JOINTS * 3}."
            f"Got {varen_model.body_pose.shape})."
        )
