import os
import pickle
import argparse
from collections import OrderedDict


def prepare_checkpoint(ckpt: OrderedDict) -> OrderedDict:
    """
    VAREN model takes a simplified model structure/naming convention than in the originally trained model.
    Specifically, the MuscleBetaPredictor and the MuscleDeformer remain the same, while the surrounding 
    code structure is simplified. To account for this, the ckpt.path file needs to be adjusted.

    The weights and biases for both the MuscleBetaPredictor and the MuscleDeformer are extracted and placed in their own sub-dictionaries inside the checkpoint.
    These are accessable through ckpt['betas_muscle_predictor'] and ckpt['Bm], respectively.
    
    """
    new_ckpt = {}
    for key in ckpt.keys():
        # rename key by removing smal. from the start if it exstis and overwrite the ckpt   
        new_key = key.replace('smal.', '')
        # if starts with Bm. then add to sub dict called Bm
        if new_key.startswith('Bm.'):
            if 'Bm' not in new_ckpt:
                new_ckpt['Bm'] = {}
            # remove Bm. from name
            new_key = new_key.replace('Bm.', '')
            new_ckpt['Bm'][new_key] = ckpt[key]
        else:
            # if starts with betas_muscle_predictor then add to sub dict called betas_muscle_predictor
            if new_key.startswith('betas_muscle_predictor.'):
                if 'betas_muscle_predictor' not in new_ckpt:
                    new_ckpt['betas_muscle_predictor'] = {}
                # remove betas_muscle_predictor. from name
                new_key = new_key.replace('betas_muscle_predictor.', '')
                new_ckpt['betas_muscle_predictor'][new_key] = ckpt[key]
            else:
                new_ckpt[new_key] = ckpt[key]
    
    return OrderedDict(new_ckpt)
        

def files_exist(files: list) -> bool:
    """
    Check if files exist.
    """
    success = True # updated in loop
    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"Warning: File not found - {file_path}.")
            success = False
    
    return success

def join_pkl_data(files: list) -> dict:
    """ 
    
    """
    new_dict = {}
    for file in files:
        with open(file, 'rb') as f:
            dat = pickle.load(f, encoding='latin1')
            new_dict.update(dat)
    return new_dict


def save_pkl(data, file_name) -> None:
    with open(file_name, 'wb') as f:
            pickle.dump(data, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/varen")
    parser.add_argument("--ckpt_name", type=str, default="pred_net_100.pth", help="Name of the checkpoint .pth file")
    args = parser.parse_args()
    
    # 1. Combine downloaded files into 1 clean VAREN.pkl
    
    # Define file paths
    base_pkl = os.path.join(args.model_dir, "varen_smal_real_horse.pkl")
    seg_data = os.path.join(args.model_dir, "varen_smal_real_horse_seg_data.pkl")

    # list of files to join
    pkl_files = [base_pkl, seg_data]

    # If files are found, process and save
    if files_exist(pkl_files):
        varen_data = join_pkl_data(pkl_files) # join data
        save_pkl(varen_data, os.path.join(args.model_dir,'VAREN.pkl'))
        print(f"Saved joined pickle data to {os.path.join(args.model_dir,'VAREN.pkl')}")
    else:
        print("Skipping pkl preparation.")
        

    # 2. Load and prepare trained model
    ckpt_path = os.path.join(args.model_dir, args.ckpt_name)
    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, weights_only=True)
        new_ckpt = prepare_checkpoint(ckpt=ckpt)
        torch.save(new_ckpt, os.path.join(args.model_dir, "varen.pth"))
        print(f"Saved cleaned VAREN checkpoint to {os.path.join(args.model_dir, 'varen.pth')}")
    else:
        print(f"Warning: File not found - {ckpt_path}. Skipping operation.")
    



if __name__ == "__main__":
    main()