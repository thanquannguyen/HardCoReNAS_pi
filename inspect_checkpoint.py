import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from timm.models import create_model
from timm.models.mobilenasnet import *

def inspect_checkpoint(checkpoint_path, mobilenet_string):
    print(f"Inspecting {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")
    
    # Check for classifier keys
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k or 'head' in k]
    print(f"Classifier keys in checkpoint: {classifier_keys}")
    for k in classifier_keys:
        print(f"  {k} shape: {state_dict[k].shape}")

    print("\nCreating model...")
    model = create_model(
        'mobilenasnet',
        pretrained=False,
        num_classes=10,
        mobilenet_string=mobilenet_string
    )
    print(f"Model keys (first 10): {list(model.state_dict().keys())[:10]}")
    
    model_classifier_keys = [k for k in model.state_dict().keys() if 'classifier' in k or 'fc' in k or 'head' in k]
    print(f"Classifier keys in model: {model_classifier_keys}")
    
    # Check for mismatches
    print("\nChecking for mismatches...")
    mismatches = []
    for k in model.state_dict().keys():
        if k not in state_dict and "module." + k not in state_dict:
             mismatches.append(k)
    
    if mismatches:
        print(f"MISSING KEYS in Checkpoint (Total {len(mismatches)}):")
        for k in mismatches[:10]:
            print(f"  {k}")
    else:
        print("All model keys found in checkpoint!")

if __name__ == "__main__":
    checkpoint_path = "outputs/fine_tuned/train/20251206-041707-mobilenasnet-224/model_best.pth.tar"
    mobilenet_string = "[['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'], ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'], ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'], ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'], ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        
    inspect_checkpoint(checkpoint_path, mobilenet_string)
