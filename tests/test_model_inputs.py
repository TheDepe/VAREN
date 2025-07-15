"""Test cases for the VAREN model inputs."""
import pytest
import torch

from varen.utils import VARENOutput


def test_warning_on_wrong_pose_key(varen_model):
    """Test if a warning is raised when an incorrect pose key is used."""
    with pytest.warns(UserWarning):
        # Using 'pose' instead of 'body_pose'
        varen_model(not_body_pose=torch.zeros((1, 37 * 3)))


def test_no_warning_on_no_input(varen_model, recwarn):
    """Test if no warning is raised when no input is provided."""
    # This should not raise any warnings
    varen_model()
    assert len(recwarn) == 0, (
        "No warnings should be raised when no input is provided."
    )


# Test moving varen model to CPU and CUDA
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_varen_model_to_cpu1(varen_model):
    """Test if the VAREN model can be moved to CPU."""
    varen_model = varen_model.to('cuda')
    varen_model = varen_model.to('cpu')
    assert varen_model.shapedirs.device == torch.device('cpu'), (
        f"Model should be on CPU, but got {varen_model.shapedirs.device}."
    )
    assert varen_model.betas.device == torch.device('cpu'), (
        f"Model should be on CPU, but got {varen_model.betas.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_varen_model_to_cpu2(varen_model):
    """Test if the VAREN model can be moved to CPU."""
    varen_model = varen_model.cuda()
    varen_model = varen_model.cpu()
    assert varen_model.shapedirs.device == torch.device('cpu'), (
        f"Model should be on CPU, but got {varen_model.shapedirs.device}."
    )
    assert varen_model.betas.device == torch.device('cpu'), (
        f"Model should be on CPU, but got {varen_model.betas.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_varen_model_to_cuda1(varen_model):
    """Test if the VAREN model can be moved to CUDA."""
    varen_model = varen_model.to('cuda')
    assert varen_model.shapedirs.is_cuda, (
        f"Vertices should be on CUDA, but got {varen_model.shapedirs.device}."
    )
    assert varen_model.betas.is_cuda, (
        f"Vertices should be on CUDA, but got {varen_model.betas.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_varen_model_to_cuda2(varen_model):
    """Test if the VAREN model can be moved to CUDA."""
    varen_model = varen_model.to('cuda')
    assert varen_model.shapedirs.is_cuda, (
        f"Vertices should be on CUDA, but got {varen_model.shapedirs.device}."
    )
    assert varen_model.betas.is_cuda, (
        f"Vertices should be on CUDA, but got {varen_model.betas.device}."
    )


# Test CPU inputs on CPU
def test_pose_cpu(varen_model, pose_cpu):
    """Test the model with CPU pose input."""
    varen_model = varen_model.to('cpu')
    output = varen_model(body_pose=pose_cpu)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.device == torch.device('cpu'), (
        f'Vertices should be on CPU, but got {output.vertices.device}.'
    )


def test_global_orient_cpu(varen_model, global_orient_cpu):
    """Test the model with CPU pose input."""
    varen_model = varen_model.to('cpu')
    output = varen_model(global_orient=global_orient_cpu)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.device == torch.device('cpu'), (
        f'Vertices should be on CPU, but got {output.vertices.device}.'
    )


def test_shape_cpu(varen_model, shape_cpu):
    """Test the model with CPU pose input."""
    varen_model = varen_model.to('cpu')
    output = varen_model(betas=shape_cpu)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.device == torch.device('cpu'), (
        f'Vertices should be on CPU, but got {output.vertices.device}.'
    )


def test_transl_cpu(varen_model, translation_cpu):
    """Test the model with CPU pose input."""
    varen_model = varen_model.to('cpu')
    output = varen_model(transl=translation_cpu)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.device == torch.device('cpu'), (
        f'Vertices should be on CPU, but got {output.vertices.device}.'
    )


# Test CUDA inputs on CUDA
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pose_cuda(varen_model, pose_cuda):
    """Test the model with CUDA pose input."""
    varen_model = varen_model.to('cuda')
    output = varen_model(body_pose=pose_cuda)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.is_cuda, (
        f"Vertices should be on CUDA, but got {output.vertices.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_orient_cuda(varen_model, global_orient_cuda):
    """Test the model with CUDA pose input."""
    varen_model = varen_model.to('cuda')
    output = varen_model(global_orient=global_orient_cuda)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.is_cuda, (
        f"Vertices should be on CUDA, but got {output.vertices.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shape_cuda(varen_model, shape_cuda):
    """Test the model with CUDA pose input."""
    varen_model = varen_model.to('cuda')
    output = varen_model(betas=shape_cuda)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.is_cuda, (
        f"Vertices should be on CUDA, but got {output.vertices.device}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transl_cuda(varen_model, translation_cuda):
    """Test the model with CUDA pose input."""
    varen_model = varen_model.to('cuda')
    output = varen_model(transl=translation_cuda)
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )
    assert output.vertices is not None, "Vertices should not be None."
    assert output.vertices.is_cuda, (
        f"Vertices should be on CUDA, but got {output.vertices.device}."
    )


# Test CUDA inputs on cpu (Should all raise an error)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pose_cuda_on_cpu(varen_model, pose_cuda):
    """Test the model with CUDA pose input on CPU."""
    varen_model = varen_model.to('cpu')
    with pytest.raises(RuntimeError):
        varen_model(body_pose=pose_cuda)


# Test CUDA inputs on cpu (Should all raise an error)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shape_cuda_on_cpu(varen_model, shape_cuda):
    """Test the model with CUDA pose input on CPU."""
    varen_model = varen_model.to('cpu')
    with pytest.raises(RuntimeError):
        varen_model(betas=shape_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_orient_cuda_on_cpu(varen_model, global_orient_cuda):
    """Test the model with CUDA global orientation input on CPU."""
    varen_model = varen_model.to('cpu')
    with pytest.raises(RuntimeError):
        varen_model(global_orient=global_orient_cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transl_cuda_on_cpu(varen_model, translation_cuda):
    """Test the model with CUDA translation input on CPU."""
    varen_model = varen_model.to('cpu')
    with pytest.raises(RuntimeError):
        varen_model(transl=translation_cuda)


# Test CPU inputs on CUDA (Should all raise an error)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pose_cpu_on_cuda(varen_model, pose_cpu):
    """Test the model with CPU pose input on CUDA."""
    varen_model = varen_model.to('cuda')
    with pytest.raises(RuntimeError):
        varen_model(body_pose=pose_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_global_orient_cpu_on_cuda(varen_model, global_orient_cpu):
    """Test the model with CPU global orientation input on CUDA."""
    varen_model = varen_model.to('cuda')
    with pytest.raises(RuntimeError):
        varen_model(global_orient=global_orient_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shape_cpu_on_cuda(varen_model, shape_cpu):
    """Test the model with CPU shape input on CUDA."""
    varen_model = varen_model.to('cuda')
    with pytest.raises(RuntimeError):
        varen_model(betas=shape_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transl_cpu_on_cuda(varen_model, translation_cpu):
    """Test the model with CPU translation input on CUDA."""
    varen_model = varen_model.to('cuda')
    with pytest.raises(RuntimeError):
        varen_model(transl=translation_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_input_missing_pose(
        varen_model,
        translation_cuda,
        global_orient_cuda,
        shape_cuda):
    """Test the model with missing pose input on CUDA. This should run."""
    varen_model = varen_model.to('cuda')
    output = varen_model(
        transl=translation_cuda,
        global_orient=global_orient_cuda,
        betas=shape_cuda
    )
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_input_missing_global_orient(
        varen_model,
        pose_cuda,
        translation_cuda,
        shape_cuda):
    """Test the model with missing global orientation input on CUDA."""
    varen_model = varen_model.to('cuda')
    output = varen_model(
        body_pose=pose_cuda,
        transl=translation_cuda,
        betas=shape_cuda
    )
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_input_missing_shape(
        varen_model,
        pose_cuda,
        global_orient_cuda,
        translation_cuda):
    """Test the model with missing shape input on CUDA. This should run."""
    varen_model = varen_model.to('cuda')
    output = varen_model(
        body_pose=pose_cuda,
        global_orient=global_orient_cuda,
        transl=translation_cuda
    )
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_input_missing_transl(
        varen_model,
        pose_cuda,
        global_orient_cuda,
        shape_cuda):
    """Test the model with missing translation input on CUDA."""
    varen_model = varen_model.to('cuda')
    output = varen_model(
        body_pose=pose_cuda,
        global_orient=global_orient_cuda,
        betas=shape_cuda
    )
    assert isinstance(output, VARENOutput), (
        "Output should be a ModelOutput instance."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_inputs(
        varen_model,
        pose_cpu,
        global_orient_cuda,
        shape_cuda,
        translation_cuda):
    """Test the model with mixed inputs.

    I.e.: (CPU pose, CUDA global orientation, CUDA shape, CUDA translation).

    """
    varen_model = varen_model.to('cuda')
    with pytest.raises(RuntimeError):
        varen_model(
        body_pose=pose_cpu,
        global_orient=global_orient_cuda,
        betas=shape_cuda,
        transl=translation_cuda
    )


def test_forward_pass_with_extreme_values(varen_model, pose_cpu, shape_cpu):
    """Test the model's forward pass with extreme input values."""
    varen_model = varen_model.cpu()  # Ensure the model is on CPU for this test
    extreme_pose = pose_cpu * 1e8  # Large values
    extreme_betas = shape_cpu * 1e8  # Large values
    output = varen_model(body_pose=extreme_pose, betas=extreme_betas)
    assert isinstance(output, VARENOutput), (
        "Output should be a VARENOutput instance."
    )
    assert torch.isfinite(output.vertices).all(), (
        "Vertices should not contain NaN or Inf values."
    )
    assert torch.isfinite(output.joints).all(), (
        "Joints should not contain NaN or Inf values."
    )


def test_output_consistency(varen_model, pose_cpu, shape_cpu):
    """Test if the model produces consistent outputs for the same inputs."""
    varen_model.cpu()
    output1 = varen_model(body_pose=pose_cpu, betas=shape_cpu)
    output2 = varen_model(body_pose=pose_cpu, betas=shape_cpu)
    assert torch.allclose(output1.vertices, output2.vertices), (
        "Vertices should be consistent across runs."
    )
    assert torch.allclose(output1.joints, output2.joints), (
        "Joints should be consistent across runs."
    )


def test_invalid_inputs(varen_model):
    """Test if the model raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        varen_model(body_pose=torch.ones((1, 100)))  # Invalid pose size
    with pytest.raises(ValueError):
        varen_model(betas=torch.ones((1, 20)))  # Invalid betas size
    with pytest.raises(ValueError):
        varen_model(global_orient=torch.ones((1, 10)))
    with pytest.raises(ValueError):
        varen_model(transl=torch.ones((1, 10)))
