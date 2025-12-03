import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model
from timm.models.mobilenasnet import *

def create_dummy_checkpoint(path, model_name='mobilenasnet', num_classes=1000, mobilenet_string=None):
    print(f"Creating dummy checkpoint for {model_name} at {path}...")
    if mobilenet_string:
        model = create_model(model_name, num_classes=num_classes, mobilenet_string=mobilenet_string)
    else:
        model = create_model(model_name, num_classes=num_classes)
        
    torch.save(model.state_dict(), path)
    print("Done.")

def main():
    os.makedirs("checkpoints", exist_ok=True)
    
    # Dummy Supernet
    if not os.path.exists("checkpoints/w_supernet.pth"):
        create_dummy_checkpoint("checkpoints/w_supernet.pth", model_name='mobilenasnet')

    # Dummy HardCoRe-NAS A
    mobilenet_string = "[['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'], ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'], ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'], ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'], ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]"
    if not os.path.exists("checkpoints/hardcorenas_a.pth"):
        create_dummy_checkpoint("checkpoints/hardcorenas_a.pth", model_name='mobilenasnet', mobilenet_string=mobilenet_string)

if __name__ == "__main__":
    main()
