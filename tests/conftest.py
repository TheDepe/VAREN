"""Fixtures and configuration for VAREN model tests."""
from pathlib import Path

import pytest
import torch

from varen import VAREN

# --------------------------------------------------------------------------- #
# SET THESE VALUES HERE
VAREN_PATH = "models/varen"  # Path to VAREN model directory
VAREN_MODEL_FILE = "VAREN.pkl"  # Name of the VAREN model file
VAREN_MUSCLE_CKPT_NAME = None  # Name of VAREN muscle checkpoint file


# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def varen_model_dir():
    """Return the path to the VAREN model checkpoint directory.

    Returns:
     VAREN_PATH (str): The path to the VAREN model checkpoint directory.

    """
    model_dir = Path(VAREN_PATH)
    if not model_dir.exists():
        raise pytest.UsageError(
            f"Checkpoint directory does not exist: {model_dir}"
            )

    if not model_dir.is_dir():
        raise pytest.UsageError(
            f"Path is not a directory: {model_dir}"
            )

    pkl_files = list(model_dir.glob("*.pkl"))

    if not pkl_files:
        raise pytest.UsageError(
            f"No .pkl files found in directory: {model_dir}"
            )

    return VAREN_PATH


@pytest.fixture(scope="session")
def varen_model_file_name(varen_model_dir):
    """Return the name of the VAREN model filename."""
    full_path = Path(varen_model_dir) / VAREN_MODEL_FILE

    if not full_path.exists():
        raise pytest.UsageError(
            f"Checkpoint file does not exist: {full_path}"
            )
    if not full_path.is_file():
        raise pytest.UsageError(
            f"Path is not a file: {full_path}"
            )

    return VAREN_MODEL_FILE


@pytest.fixture(scope="session")
def ckpt_file(varen_model_dir):
    """Return the name of the VAREN muscle checkpoint file."""
    if VAREN_MUSCLE_CKPT_NAME is None:
        return None

    full_path = Path(varen_model_dir) / VAREN_MUSCLE_CKPT_NAME

    if not full_path.exists():
        raise pytest.UsageError(
            f"Checkpoint file does not exist: {full_path}"
            )
    if not full_path.is_file():
        raise pytest.UsageError(
            f"Path is not a file: {full_path}"
            )
    return VAREN_MUSCLE_CKPT_NAME


@pytest.fixture(scope="module")
def varen_model_no_muscles(varen_model_dir, varen_model_file_name):
    """Return a VAREN model instance."""
    varen_pkl = Path(varen_model_file_name)
    path, ext = varen_pkl.stem, varen_pkl.suffix

    return VAREN(
        model_path=varen_model_dir,
        model_file_name=path,
        ext=ext,
        use_muscle_deformations=False
    )


@pytest.fixture(scope="module")
def varen_model(varen_model_dir, varen_model_file_name, ckpt_file):
    """Return a VAREN model instance."""
    varen_pkl = Path(varen_model_file_name)
    path, ext = varen_pkl.stem, varen_pkl.suffix

    return VAREN(
        model_path=varen_model_dir,
        model_file_name=path,
        ext=ext,
        use_muscle_deformations=True,
        ckpt_file=ckpt_file
    )


@pytest.fixture
def pose_cpu():
    """Return a dummy pose tensor for testing."""
    return torch.zeros((1, 37 * 3))


@pytest.fixture
def pose_cuda(pose_cpu):
    """Return a dummy pose tensor on CUDA."""
    return pose_cpu.cuda()


@pytest.fixture
def shape_cpu():
    """Return a dummy shape tensor for testing."""
    return torch.zeros((1, 39))


@pytest.fixture
def shape_cuda(shape_cpu):
    """Return a dummy shape tensor on CUDA."""
    return shape_cpu.cuda()


@pytest.fixture
def global_orient_cpu():
    """Return a dummy global orientation tensor for testing."""
    return torch.zeros((1, 3))


@pytest.fixture
def global_orient_cuda(global_orient_cpu):
    """Return a dummy global orientation tensor on CUDA."""
    return global_orient_cpu.cuda()


@pytest.fixture
def translation_cpu():
    """Return a dummy translation tensor for testing."""
    return torch.ones((1, 3))


@pytest.fixture
def translation_cuda(translation_cpu):
    """Return a dummy translation tensor on CUDA."""
    return translation_cpu.cuda()
