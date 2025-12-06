import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import argparse
import sys
import os

# Add project root to path to import timm and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timm.models import create_model
from timm.models.mobilenasnet import * # Register models
import low_rank
import quantize_onnx
import torch.nn.utils.prune as prune

def get_train_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Warning: Train data not found at {train_dir}")
        return None
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def fine_tune(model, teacher_model, train_loader, epochs=1, lr=0.001, device='cpu', max_batches=None):
    """
    Fine-tune the model (Healing) with Knowledge Distillation.
    """
    print(f"Fine-tuning model for {epochs} epochs (LR={lr})...")
    model.to(device)
    model.train()
    
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()
        print("Using Knowledge Distillation.")
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if max_batches and i >= max_batches:
                print(f"  Reached max_batches ({max_batches}). Stopping epoch early.")
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            # KD Loss
            if teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # SoftTarget Loss (KLDiv)
                T = 4.0
                alpha = 0.9
                soft_loss = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(outputs/T, dim=1),
                    F.softmax(teacher_outputs/T, dim=1)
                ) * (T * T)
                
                loss = loss * (1. - alpha) + soft_loss * alpha
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 20 == 0: # Print more frequently for short runs
                print(f"  Epoch {epoch+1}, Batch {i}, Loss: {running_loss/20:.4f}")
                running_loss = 0.0
    
    model.eval()
    return model

def apply_pruning(model, amount=0.2):
    """
    Apply L1 Unstructured Pruning to all Conv2d layers.
    """
    print(f"Pruning model with amount={amount}...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Do NOT remove pruning hooks yet if we want to fine-tune with masks?
            # Actually, for simple unstructured pruning, we can make it permanent 
            # but usually we train with the mask. 
            # For simplicity here, we make it permanent immediately so we don't deal with re-parametrization issues during save/load.
            # But ideally: Prune -> Train (Masked) -> Remove.
            # Here: Prune -> Remove -> Train (Weights are 0, but gradients might update them? No, remove makes them 0 hard).
            # Wait, prune.remove makes the pruning permanent (applies mask to weight). 
            # If we train after prune.remove, the 0 weights CAN become non-zero again if gradient is non-zero.
            # So we should NOT call prune.remove() before fine-tuning if we want to keep sparsity.
            # But standard PyTorch pruning handles this by registering a hook.
            pass 
    return model

def make_pruning_permanent(model):
    print("Making pruning permanent...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
    return model

def run_compression(args):
    print("Starting ADVANCED compression pipeline...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Create Teacher Model for KD
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    
    # Prepare Data
    train_loader = get_train_loader('data/cifar10_images', batch_size=32)
    
    model.eval()
    
    # 2. Pruning
    if args.prune:
        model = apply_pruning(model, amount=args.prune_amount)
        # Fine-tune to heal
        if train_loader and not args.compressed_checkpoint:
            model = fine_tune(model, teacher_model, train_loader, epochs=1, lr=0.001, device=device, max_batches=100)
        # Make permanent
        model = make_pruning_permanent(model)
        
    # 3. Low-Rank Factorization
    if args.low_rank:
        model = low_rank.apply_low_rank(model, rank_ratio=args.rank_ratio)
        # Fine-tune to heal
        if train_loader and not args.compressed_checkpoint:
            model = fine_tune(model, teacher_model, train_loader, epochs=1, lr=0.001, device=device, max_batches=100)
        
    # Load compressed weights if provided (Late Loading)
    if args.compressed_checkpoint:
        print(f"Loading compressed weights from {args.compressed_checkpoint}")
        checkpoint = torch.load(args.compressed_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle potential key mismatches
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded compressed weights.")
        
    # Save intermediate pytorch model
    if not args.compressed_checkpoint:
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
            reader = quantize_onnx.CIFAR10CalibrationDataReader(val_dir, count=100)
            print("Performing Static Quantization (Entropy, Per-Channel)...")
            quantize_onnx.quantize_static(onnx_path, quantized_path, reader,
                            weight_type=quantize_onnx.QuantType.QInt8,
                            activation_type=quantize_onnx.QuantType.QUInt8,
                            calibrate_method=quantize_onnx.CalibrationMethod.Entropy,
                            per_channel=True)
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
    parser.add_argument('--compressed_checkpoint', type=str, default='', help='Path to already compressed checkpoint (for quantization only)')
    
    args = parser.parse_args()
    
    run_compression(args)
