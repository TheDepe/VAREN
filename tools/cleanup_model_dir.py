"""Tool for deleting or moving old model file structure."""
import os
import shutil


def handle_files(args):
    """Remove and/or move old files from model directory."""
    # Define file paths
    base_pkl = os.path.join(
        args.model_dir, "varen_smal_real_horse.pkl"
        )
    seg_data = os.path.join(
        args.model_dir, "varen_smal_real_horse_seg_data.pkl"
        )
    ckpt_path = os.path.join(args.model_dir, args.ckpt_name)

    files_to_handle = [base_pkl, seg_data, ckpt_path]

    # Create a folder for original data if moving files
    destination_folder = os.path.join(args.model_dir, ".original_data")

    # Prompt the user to choose between deletion or moving

    if args.delete:
        confirm = input(
            "Are you sure you want to delete these files? This action is \
            irreversible. To move instead, select 'n' (y/n): "
            ).strip().lower()

    if args.delete and confirm == "y":
        for file_path in files_to_handle:
            if os.path.isfile(file_path):
                os.remove(file_path)

    else:

        for file_path in files_to_handle:
            if os.path.isfile(file_path):
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                dest_path = os.path.join(
                    destination_folder, os.path.basename(file_path)
                    )
                shutil.move(file_path, dest_path)
                print(f"Moved: {file_path} -> {dest_path}")
            else:
                print(f"File not found {file_path}")

    print("Directory has been cleaned.")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete or move specific files."
        )
    parser.add_argument("--model_dir", type=str, default="models/varen")
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="pred_net_100.pth",
        help="Name of the checkpoint .pth file"
        )
    parser.add_argument(
        '--delete',
        action='store_true',
        help="Delete the excess files"
        )
    args = parser.parse_args()

    handle_files(args)
