"""Test cases for loading VAREN models / IO."""
from pathlib import Path

import pytest

from varen import VAREN


def test_loading_without_muscles(varen_model_no_muscles):
    """Test to see if the model loads without muscles."""
    varen = varen_model_no_muscles
    assert isinstance(varen, VAREN), "Model is not an instance of VAREN"


def test_loading(varen_model):
    """Test to see if the model loads with muscles."""
    varen = varen_model
    assert isinstance(varen, VAREN), "Model is not an instance of VAREN"


def test_varen_model_loading_bad_model_dir(varen_model_file_name
    ):
    """Test to see if model fails to load when bad dir is passed."""
    bad_model_dir = "/non/existent/path"

    varen_pkl = Path(varen_model_file_name)
    path, ext = varen_pkl.stem, varen_pkl.suffix

    with pytest.raises(FileNotFoundError):
        VAREN(
            model_path=bad_model_dir,
            model_file_name=path,
            use_muscle_deformations=False,
            ext=ext
        )


def test_varen_model_loading_bad_model_filename(
        varen_model_dir,
        varen_model_file_name
        ):
    """Test to see if model fails to load when bad filename is passed."""
    bad_model_file_name = "missing_model_45812187"

    varen_pkl = Path(varen_model_file_name)
    _, ext = varen_pkl.stem, varen_pkl.suffix
    with pytest.raises(FileNotFoundError):
        VAREN(
            model_path=varen_model_dir,
            model_file_name=bad_model_file_name,
            use_muscle_deformations=False,
            ext=ext
        )


def test_varen_model_loading_bad_model_ext(
        varen_model_dir, varen_model_file_name
    ):
    """Test to see if model fails to load when bad filename is passed."""
    varen_pkl = Path(varen_model_file_name)
    path, _ = varen_pkl.stem, varen_pkl.suffix

    with pytest.raises(FileNotFoundError):
        VAREN(
            model_path=varen_model_dir,
            model_file_name=path,
            use_muscle_deformations=False,
            ext="notarealext"
        )


def test_varen_model_loading_bad_ckpt_filename(
        varen_model_dir, varen_model_file_name
    ):
    """Test to see if model fails to load when bad filename is passed."""
    varen_pkl = Path(varen_model_file_name)
    path, ext = varen_pkl.stem, varen_pkl.suffix

    with pytest.raises(FileNotFoundError):
        VAREN(
            model_path=varen_model_dir,
            model_file_name=path,
            use_muscle_deformations=True,
            ext=ext,
            ckpt_file="/non/existent/path/to/ckpt.pth"
        )
