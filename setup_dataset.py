import os
import shutil
from tqdm import tqdm

def setup_training_data():
    # Base directory where you saved the dataset
    source_dir = r"E:\dataset_heart\archive\chest_xray"
    
    # Create our dataset structure
    base_dir = 'dataset'
    for category in ['train_images', 'val_images']:
        for condition in ['normal', 'pneumonia']:
            os.makedirs(os.path.join(base_dir, category, condition), exist_ok=True)

    # Process each split (train, val, test)
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\nProcessing {split} set...")
        
        # Source directories
        normal_src = os.path.join(source_dir, split, 'NORMAL')
        pneumonia_src = os.path.join(source_dir, split, 'PNEUMONIA')
        
        # Destination split (combine test into val for simplicity)
        dest_split = 'val_images' if split == 'test' else 'train_images'
        
        # Copy normal images
        if os.path.exists(normal_src):
            normal_images = os.listdir(normal_src)
            print(f"Copying {len(normal_images)} normal images...")
            for img in tqdm(normal_images):
                src = os.path.join(normal_src, img)
                dst = os.path.join(base_dir, dest_split, 'normal', img)
                shutil.copy2(src, dst)
        
        # Copy pneumonia images
        if os.path.exists(pneumonia_src):
            pneumonia_images = os.listdir(pneumonia_src)
            print(f"Copying {len(pneumonia_images)} pneumonia images...")
            for img in tqdm(pneumonia_images):
                src = os.path.join(pneumonia_src, img)
                dst = os.path.join(base_dir, dest_split, 'pneumonia', img)
                shutil.copy2(src, dst)

    # Print statistics
    print("\nDataset Statistics:")
    for split in ['train_images', 'val_images']:
        print(f"\n{split}:")
        for condition in ['normal', 'pneumonia']:
            path = os.path.join(base_dir, split, condition)
            num_images = len(os.listdir(path))
            print(f"{condition}: {num_images} images")

if __name__ == "__main__":
    setup_training_data() 