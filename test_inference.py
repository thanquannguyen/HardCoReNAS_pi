import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from timm.models import create_model
from timm.models.mobilenasnet import *

def main():
    mobilenet_string = "[['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'], ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'], ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'], ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'], ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]"
    
    print("Creating model...")
    model = create_model('mobilenasnet', mobilenet_string=mobilenet_string)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {input_tensor.shape}")
    
    print("Running inference...")
    try:
        out = model(input_tensor)
        print(f"Output shape: {out.shape}")
        print("Inference successful.")
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
