import argparse
import time
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model
from timm.models.mobilenasnet import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def measure_latency_pytorch(model, input_shape=(1, 3, 224, 224), repeats=100):
    model.eval()
    input_data = torch.randn(input_shape)
    if torch.cuda.is_available():
        model = model.cuda()
        input_data = input_data.cuda()
        
    # Warmup
    for _ in range(10):
        _ = model(input_data)
        
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(input_data)
    end = time.time()
    
    avg_latency = (end - start) / repeats * 1000 # ms
    return avg_latency

def measure_latency_onnx(session, input_shape=(1, 3, 224, 224), repeats=100):
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})
        
    start = time.time()
    for _ in range(repeats):
        _ = session.run(None, {input_name: input_data})
    end = time.time()
    
    avg_latency = (end - start) / repeats * 1000 # ms
    return avg_latency

def evaluate_accuracy(model_or_session, val_dir, is_onnx=False, batch_size=32):
    print(f"Evaluating accuracy on {val_dir}...")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    correct = 0
    total = 0
    
    # Check if ONNX
    if is_onnx:
        input_name = model_or_session.get_inputs()[0].name
        
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            if is_onnx:
                # ONNX Runtime expects numpy
                inputs = images.numpy()
                outputs = model_or_session.run(None, {input_name: inputs})
                # outputs is a list, take the first element
                outputs = torch.from_numpy(outputs[0])
            else:
                # PyTorch
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                    model_or_session = model_or_session.cuda()
                outputs = model_or_session(images)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def main(args):
    print(f"Benchmarking {args.model_path}...")
    
    is_onnx = args.model_path.endswith('.onnx')
    
    if is_onnx:
        session = ort.InferenceSession(args.model_path)
        latency = measure_latency_onnx(session)
        model_size = os.path.getsize(args.model_path) / 1024 / 1024 # MB
        print(f"Model: ONNX")
    else:
        # Load PyTorch model
        # We need the string to build it
        if not args.mobilenet_string:
            print("Error: --mobilenet_string required for PyTorch model")
            return
            
        model = create_model(
            'mobilenasnet',
            pretrained=False,
            num_classes=args.num_classes,
            mobilenet_string=args.mobilenet_string
        )
        checkpoint = torch.load(args.model_path, map_location='cpu')
        # Load state dict logic...
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        
        latency = measure_latency_pytorch(model)
        model_size = os.path.getsize(args.model_path) / 1024 / 1024 # MB
        print(f"Model: PyTorch")

    print(f"Latency: {latency:.2f} ms")
    print(f"Size: {model_size:.2f} MB")
    
    # Accuracy evaluation
    val_dir = "data/cifar10_images/val"
    if os.path.exists(val_dir):
        if is_onnx:
            acc = evaluate_accuracy(session, val_dir, is_onnx=True)
        else:
            acc = evaluate_accuracy(model, val_dir, is_onnx=False)
        print(f"Accuracy (Top-1): {acc:.2f}%")
    else:
        print(f"Validation data not found at {val_dir}. Skipping accuracy evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HardCoReNAS Benchmark')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth or .onnx file')
    parser.add_argument('--mobilenet_string', type=str, default='', help='Architecture string (for PyTorch)')
    parser.add_argument('--num_classes', type=int, default=1000)
    args = parser.parse_args()
    
    main(args)
