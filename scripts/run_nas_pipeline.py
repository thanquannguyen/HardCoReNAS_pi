import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def main():
    # Configuration
    dataset_path = "data/cifar10" # Placeholder
    lut_file = "lut_cpu.pkl"
    
    # 1. Measure Latency LUT
    print("--- Step 1: Measure Latency LUT ---")
    # Assuming we run on CPU for now as proxy or if on Pi
    run_command(f"python measure_latency_lut.py --target_device=cpu --lut_filename={lut_file}")
    
    # 2. Search (Using downloaded Supernet)
    print("--- Step 2: Search Architecture ---")
    # We use the downloaded supernet checkpoint
    supernet_ckpt = "checkpoints/w_supernet.pth"
    if os.path.exists(supernet_ckpt):
        # Run search with a loose constraint for demo purposes
        # Note: dataset_path needs to be correct. We use data/cifar10_images
        cmd = f"python search.py data/cifar10_images --lut_filename={lut_file} --inference_time_limit=50 --initial-checkpoint={supernet_ckpt} --train_percent=10 --bcfw_steps=100"
        # run_command(cmd) # Uncomment to run actual search (takes time)
        print("Search command prepared (commented out for speed):")
        print(cmd)
    else:
        print(f"Supernet checkpoint {supernet_ckpt} not found. Run scripts/download_checkpoints.py first.")
    
    # Demo Architecture (HardCoRe-NAS A)
    mobilenet_string = "[['ds_r1_k3_s1_e1_c16_nre'], ['ir_r1_k5_s2_e3_c24_nre', 'ir_r1_k5_s1_e3_c24_nre_se0.25'], ['ir_r1_k5_s2_e3_c40_nre', 'ir_r1_k5_s1_e6_c40_nre_se0.25'], ['ir_r1_k5_s2_e6_c80_se0.25', 'ir_r1_k5_s1_e6_c80_se0.25'], ['ir_r1_k5_s1_e6_c112_se0.25', 'ir_r1_k5_s1_e6_c112_se0.25'], ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25'], ['cn_r1_k1_s1_c960']]"
    
    # 3. Train (Fine-tune)
    print("--- Step 3: Fine-tune (Real Training) ---")
    # We use the ImageNet checkpoint as a starting point (Transfer Learning)
    pretrained_ckpt = "checkpoints/hardcorenas_a.pth"
    output_dir = "outputs/fine_tuned"
    
    # Command for fine-tuning on CIFAR-10
    # We set epochs=5 for a quick demo. Increase to 50+ for high accuracy.
    train_cmd = (
        f"python train.py data/cifar10_images "
        f"--mobilenet_string=\"{mobilenet_string}\" "
        f"--num-classes 10 "
        f"--initial-checkpoint {pretrained_ckpt} "
        f"--epochs 100 "
        f"--batch-size 32 "
        f"--output {output_dir} "
        f"--validation-batch-size-multiplier 1 "
        f"--no-prefetcher" 
    )
    
    if os.path.exists(pretrained_ckpt):
        run_command(train_cmd)
        # Update checkpoint path for compression to use the fine-tuned model
        checkpoint = f"{output_dir}/model_best.pth.tar"
    else:
        print("Pretrained checkpoint not found. Skipping training.")
        checkpoint = pretrained_ckpt # Fallback

    # 4. Compression Pipeline
    print("--- Step 4: Compression ---")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found. Skipping compression.")
    else:
        # Run compression pipeline with num_classes=10
        run_command(f"python compression/compress_pipeline.py --mobilenet_string=\"{mobilenet_string}\" --checkpoint={checkpoint} --num_classes=10 --prune --low_rank --quantize")

if __name__ == "__main__":
    main()
