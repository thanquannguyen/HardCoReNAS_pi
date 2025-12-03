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

# Helper for calibration data (if needed for static quantization)
class RandomCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_shape, count=10):
        self.input_shape = input_shape
        self.count = count
        self.current = 0
        
    def get_next(self):
        if self.current >= self.count:
            return None
        self.current += 1
        return {'input': np.random.randn(1, *self.input_shape).astype(np.float32)}

import numpy as np
