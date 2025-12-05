import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data_batch, output_dir, label_names):
    images = data_batch[b'data']
    labels = data_batch[b'labels']
    filenames = data_batch[b'filenames']
    
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    for i in tqdm(range(len(images))):
        label = labels[i]
        label_name = label_names[label].decode('utf-8')
        image_data = images[i]
        filename = filenames[i].decode('utf-8')
        
        class_dir = os.path.join(output_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)
        
        img = Image.fromarray(image_data)
        img.save(os.path.join(class_dir, filename))

def main():
    input_dir = 'data/cifar10'
    output_dir = 'data/cifar10_images'
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found. Downloading CIFAR-10...")
        import urllib.request
        import tarfile
        
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        
        # Download
        try:
            urllib.request.urlretrieve(url, filename)
            print("Download complete. Extracting...")
            
            # Extract
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall("data")
            
            # Rename folder if necessary (tar extracts to cifar-10-batches-py)
            extracted_dir = "data/cifar-10-batches-py"
            if os.path.exists(extracted_dir):
                if os.path.exists(input_dir):
                    import shutil
                    shutil.rmtree(input_dir)
                os.rename(extracted_dir, input_dir)
                
            print("Extraction complete.")
            os.remove(filename)
            
        except Exception as e:
            print(f"Error downloading/extracting CIFAR-10: {e}")
            return

    # Load meta
    meta_file = os.path.join(input_dir, 'batches.meta')
    if not os.path.exists(meta_file):
        print("batches.meta not found. Retrying download...")
        import shutil
        shutil.rmtree(input_dir)
        main() # Retry
        return
        
    try:
        meta = unpickle(meta_file)
    except Exception as e:
        print(f"Error unpickling meta file: {e}. Data might be corrupted (LFS pointer?). Deleting and retrying...")
        import shutil
        shutil.rmtree(input_dir)
        main() # Retry
        return

    label_names = meta[b'label_names']
    
    # Process Train
    train_dir = os.path.join(output_dir, 'train')
    print("Processing Train batches...")
    for i in range(1, 6):
        batch_file = os.path.join(input_dir, f'data_batch_{i}')
        try:
            data = unpickle(batch_file)
            save_images(data, train_dir, label_names)
        except Exception as e:
             print(f"Error processing batch {i}: {e}. Retrying...")
             import shutil
             shutil.rmtree(input_dir)
             main()
             return
        
    # Process Test (Val)
    val_dir = os.path.join(output_dir, 'val')
    print("Processing Test batch...")
    test_file = os.path.join(input_dir, 'test_batch')
    data = unpickle(test_file)
    save_images(data, val_dir, label_names)
    
    print("Conversion complete!")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    main()
