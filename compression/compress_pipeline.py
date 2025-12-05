import sys
import os
import argparse
import torch
import torch.nn as nn
# Add project root to path to import timm and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timm.models import create_model
from timm.models.mobilenasnet import * # Register models
import low_rank
import quantize_onnx
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.2):
    """
    Apply L1 Unstructured Pruning to all Conv2d layers.
    """
    print(f"Pruning model with amount={amount}...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight') # Make pruning permanent
    return model

def run_compression(args):
    print("Starting compression pipeline...")
    
    # 1. Load Model
    print(f"Creating model with string: {args.mobilenet_string}")
    model = create_model(
        'mobilenasnet',
        pretrained=False,
        num_classes=args.num_classes,
        mobilenet_string=args.mobilenet_string
    )
    
    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Handle potential key mismatches (e.g. module. prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    # 2. Pruning
    if args.prune:
        model = apply_pruning(model, amount=args.prune_amount)
        
    # 3. Low-Rank Factorization
    if args.low_rank:
        model = low_rank.apply_low_rank(model, rank_ratio=args.rank_ratio)
        
    # Save intermediate pytorch model
    torch.save(model.state_dict(), "compressed_model.pth")
    print("Saved compressed PyTorch model to compressed_model.pth")

    # 4. Quantization (via ONNX)
    if args.quantize:
        input_shape = (1, 3, 224, 224) # Default input size
        onnx_path = "model.onnx"
        quantized_path = "model_quantized.onnx"
        
        quantize_onnx.export_onnx(model, input_shape, onnx_path)
        
        # Use Static Quantization with Calibration
        print("Preparing calibration data...")
        val_dir = "data/cifar10_images/val"
        if os.path.exists(val_dir):
            reader = quantize_onnx.CIFAR10CalibrationDataReader(val_dir, count=300)
            quantize_onnx.quantize_model(onnx_path, quantized_path, calibration_data_reader=reader)
        else:
            print(f"Warning: {val_dir} not found. Falling back to Dynamic Quantization (Not recommended for CNNs).")
            quantize_onnx.quantize_model(onnx_path, quantized_path)

    print("Compression finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HardCoReNAS Compression Pipeline')
    parser.add_argument('--mobilenet_string', type=str, required=True, help='Architecture string')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to input checkpoint')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
    
    parser.add_argument('--prune', action='store_true', help='Apply pruning')
    parser.add_argument('--prune_amount', type=float, default=0.2, help='Pruning amount')
    
    parser.add_argument('--low_rank', action='store_true', help='Apply low-rank factorization')
    parser.add_argument('--rank_ratio', type=float, default=0.5, help='Rank ratio for SVD')
    
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    
    args = parser.parse_args()
    
    run_compression(args)
