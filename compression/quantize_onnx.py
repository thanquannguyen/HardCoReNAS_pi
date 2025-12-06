import torch
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType

def export_onnx(model, input_shape, onnx_path):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    print(f"Exporting ONNX with input shape: {dummy_input.shape}")
    torch.onnx.export(model, dummy_input, onnx_path, 
                      opset_version=11, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model exported to {onnx_path}")

def quantize_model(onnx_path, quantized_path, calibration_data_reader=None):
    """
    Quantize ONNX model.
    If calibration_data_reader is provided, use Static Quantization.
    Else, use Dynamic Quantization.
    """
    if calibration_data_reader:
        print("Performing Static Quantization...")
        quantize_static(onnx_path, quantized_path, calibration_data_reader)
    else:
        print("Performing Dynamic Quantization...")
        quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    
    print(f"Quantized model saved to {quantized_path}")

import os
import numpy as np
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader

import torchvision.transforms as transforms

class CIFAR10CalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir, input_name='input', count=100):
        self.data_dir = data_dir
        self.input_name = input_name
        self.count = count
        self.image_paths = []
        
        # Collect ALL image paths first to ensure random sampling across classes
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        import random
        random.shuffle(self.image_paths)
        
        # Limit to count
        if len(self.image_paths) > self.count:
            self.image_paths = self.image_paths[:self.count]
        
        self.enum_data = iter(self.image_paths)
        
        # Exact transform from evaluate.py
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_next(self):
        try:
            image_path = next(self.enum_data)
            img = Image.open(image_path).convert('RGB')
            
            # Apply torchvision transform
            img_tensor = self.transform(img)
            
            # Convert to numpy and add batch dimension
            norm_img_data = img_tensor.numpy().reshape(1, 3, 224, 224)
            
            return {self.input_name: norm_img_data}
        except StopIteration:
            return None

    def __len__(self):
        return self.count
