import torch
import sys
import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from timm.models import create_model
from timm.models.mobilenasnet import *

# CIFAR-10 Classes
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    
    # Preprocessing for the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(img).unsqueeze(0)
    return img, input_tensor

def main():
    parser = argparse.ArgumentParser(description='HardCoReNAS Inference Test')
    parser.add_argument('image_path', nargs='?', help='Path to input image')
    args = parser.parse_args()

    mobilenet_string = "[['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'], ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'], ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'], ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'], ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]"
    
    # 1. Load Model
    print("Loading model...")
    
    # Check for ONNX model first (Deployment scenario)
    onnx_path = "model_quantized.onnx"
    if os.path.exists(onnx_path):
        print(f"Loading ONNX model from {onnx_path}")
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        is_onnx = True
    else:
        # Fallback to PyTorch model
        is_onnx = False
        model = create_model('mobilenasnet', num_classes=10, mobilenet_string=mobilenet_string) # num_classes=10 for CIFAR-10
        
        # Load weights (prefer compressed_model.pth, then fine_tuned, then initial)
        if os.path.exists("compressed_model.pth"):
            checkpoint_path = "compressed_model.pth"
        elif os.path.exists("outputs/fine_tuned/model_best.pth.tar"):
            checkpoint_path = "outputs/fine_tuned/model_best.pth.tar"
        else:
            checkpoint_path = "checkpoints/hardcorenas_a.pth"
            
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
        
        model.eval()
    
    # 2. Select Image
    if args.image_path:
        image_path = args.image_path
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        print(f"Testing with provided image: {image_path}")
        random_class = "Unknown"
    else:
        # Pick a random image from validation set
        val_dir = "data/cifar10_images/val"
        if not os.path.exists(val_dir):
            print(f"Validation directory {val_dir} not found.")
            return

        # Get all class folders
        class_folders = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if not class_folders:
            print("No class folders found.")
            return
            
        # Pick random class and image
        random_class = random.choice(class_folders)
        class_path = os.path.join(val_dir, random_class)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg'))]
        
        if not images:
            print(f"No images found in {class_path}")
            return
            
        image_name = random.choice(images)
        image_path = os.path.join(class_path, image_name)
        print(f"Testing with random image: {image_path} (True Label: {random_class})")
    
    # 3. Run Inference
    original_img, input_tensor = load_image(image_path)
    
    if is_onnx:
        input_name = session.get_inputs()[0].name
        # ONNX Runtime expects numpy
        inputs = input_tensor.numpy()
        outputs = session.run(None, {input_name: inputs})
        # outputs is a list, take the first element
        output_tensor = torch.from_numpy(outputs[0])
        probabilities = F.softmax(output_tensor, dim=1)
    else:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
        
    # Get Top-1 Prediction
    # Note: The model was initialized with num_classes=1000 (ImageNet default) for the dummy checkpoint
    # But we are testing on CIFAR-10. Since we used a dummy checkpoint or a pre-trained ImageNet one,
    # the predictions might not map directly to CIFAR-10 classes correctly without fine-tuning.
    # However, for visualization purposes, we will show the predicted index and probability.
    
    top_prob, top_catid = torch.topk(probabilities, 1)
    predicted_idx = top_catid.item()
    confidence = top_prob.item() * 100
    
    print(f"Prediction: Class Index {predicted_idx}, Confidence: {confidence:.2f}%")
    
    # 4. Visualize
    draw = ImageDraw.Draw(original_img)
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    text = f"Pred: {predicted_idx}\nConf: {confidence:.1f}%"
    
    # Draw text with outline for visibility
    x, y = 5, 5
    draw.text((x-1, y), text, font=font, fill="black")
    draw.text((x+1, y), text, font=font, fill="black")
    draw.text((x, y-1), text, font=font, fill="black")
    draw.text((x, y+1), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill="white")
    
    # Resize for better visibility if it's small (CIFAR is 32x32)
    vis_img = original_img.resize((256, 256), resample=Image.NEAREST)
    
    output_file = "inference_result.png"
    vis_img.save(output_file)
    print(f"Result saved to {output_file}")

if __name__ == "__main__":
    main()
