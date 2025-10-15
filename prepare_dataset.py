import os
import pandas as pd
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Paths (adjust to your extracted dataset locations)
fer_base_path = 'dataset/fer/'  # Your extracted FER-2013 folder
nthu_base_path = 'dataset/nthu/'  # Your extracted NTHU-DDD folder
output_dir = 'dataset/'
combined_dir = os.path.join(output_dir, 'combined/')
os.makedirs(combined_dir, exist_ok=True)

# Configuration
TARGET_SIZE = (48, 48)
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)  # Use CPU cores efficiently
BALANCE_CLASSES = True  # Balance classes to prevent overfitting
MAX_SAMPLES_PER_CLASS = 5000  # Limit samples per class (adjust as needed)

# Define subdirectories and labels
fer_subdirs = {'sad': 1, 'fear': 1, 'angry': 1, 'neutral': 0, 'happy' : 0, 'disgust': 1}  # FER-2013 labels
nthu_subdirs = {'drowsy': 1, 'alert': 0}  # NTHU-DDD labels

print("=" * 60)
print("OPTIMIZED DATASET PREPARATION")
print("=" * 60)

# Function to process a single image
def process_image(args):
    """Process and save a single image with error handling"""
    file_path, output_path, label = args
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize with best interpolation
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Optional: Apply histogram equalization for better contrast
        img = cv2.equalizeHist(img)
        
        cv2.imwrite(output_path, img)
        return [output_path.replace(output_dir, ''), label]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to collect files with sampling
def collect_files(base_path, subdirs, max_samples):
    """Collect files from subdirectories with optional sampling"""
    files_dict = {}
    for subdir, label in subdirs.items():
        subdir_path = os.path.join(base_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Directory not found: {subdir_path}")
            files_dict[subdir] = []
            continue
        
        files = [os.path.join(subdir_path, f) 
                for f in os.listdir(subdir_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Sample if too many files
        if max_samples and len(files) > max_samples:
            np.random.seed(42)  # For reproducibility
            files = list(np.random.choice(files, max_samples, replace=False))
            print(f"  {subdir}: Sampled {len(files)} from total files")
        else:
            print(f"  {subdir}: {len(files)} files")
        
        files_dict[subdir] = files
    return files_dict

# Collect FER-2013 files
print("\nCollecting FER-2013 files...")
fer_files = collect_files(fer_base_path, fer_subdirs, MAX_SAMPLES_PER_CLASS)

# Collect NTHU-DDD files
print("\nCollecting NTHU-DDD files...")
nthu_files = collect_files(nthu_base_path, nthu_subdirs, MAX_SAMPLES_PER_CLASS)

# Balance classes if enabled
if BALANCE_CLASSES:
    print("\nBalancing classes...")
    # Count samples per label
    label_counts = {0: 0, 1: 0}
    for subdir, files in fer_files.items():
        label_counts[fer_subdirs[subdir]] += len(files)
    for subdir, files in nthu_files.items():
        label_counts[nthu_subdirs[subdir]] += len(files)
    
    print(f"  Before balancing - Label 0: {label_counts[0]}, Label 1: {label_counts[1]}")
    
    # Find minimum count
    min_count = min(label_counts.values())
    
    # Balance FER-2013
    for subdir in fer_files:
        label = fer_subdirs[subdir]
        if len(fer_files[subdir]) > min_count // 2:
            np.random.seed(42)
            fer_files[subdir] = list(np.random.choice(
                fer_files[subdir], 
                min(len(fer_files[subdir]), min_count // 2), 
                replace=False
            ))
    
    # Balance NTHU-DDD
    for subdir in nthu_files:
        label = nthu_subdirs[subdir]
        if len(nthu_files[subdir]) > min_count // 2:
            np.random.seed(42)
            nthu_files[subdir] = list(np.random.choice(
                nthu_files[subdir], 
                min(len(nthu_files[subdir]), min_count // 2), 
                replace=False
            ))
    
    # Recount
    label_counts = {0: 0, 1: 0}
    for subdir, files in fer_files.items():
        label_counts[fer_subdirs[subdir]] += len(files)
    for subdir, files in nthu_files.items():
        label_counts[nthu_subdirs[subdir]] += len(files)
    print(f"  After balancing - Label 0: {label_counts[0]}, Label 1: {label_counts[1]}")

# Prepare processing tasks
print("\nPreparing processing tasks...")
tasks = []
combined_start_idx = 0

# FER-2013 tasks
for subdir, files in fer_files.items():
    for j, file_path in enumerate(files):
        new_name = f'combined_{combined_start_idx + j}.jpg'
        output_path = os.path.join(combined_dir, new_name)
        tasks.append((file_path, output_path, fer_subdirs[subdir]))
    combined_start_idx += len(files)

# NTHU-DDD tasks
for subdir, files in nthu_files.items():
    for j, file_path in enumerate(files):
        new_name = f'combined_{combined_start_idx + j}.jpg'
        output_path = os.path.join(combined_dir, new_name)
        tasks.append((file_path, output_path, nthu_subdirs[subdir]))
    combined_start_idx += len(files)

print(f"Total images to process: {len(tasks)}")

# Process images in parallel with progress bar
print(f"\nProcessing images with {MAX_WORKERS} workers...")
all_files = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_image, task): task for task in tasks}
    
    with tqdm(total=len(tasks), desc="Processing", unit="img") as pbar:
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_files.append(result)
            pbar.update(1)

# Shuffle the dataset for better training
print("\nShuffling dataset...")
np.random.seed(42)
np.random.shuffle(all_files)

# Create labels.csv
labels_df = pd.DataFrame(all_files, columns=['image_path', 'label'])
csv_path = os.path.join(output_dir, 'labels.csv')
labels_df.to_csv(csv_path, index=False)

# Print statistics
print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)
print(f"Total images processed: {len(all_files)}")
print(f"Label distribution:")
print(labels_df['label'].value_counts().sort_index())
print(f"\nClass balance ratio: {labels_df['label'].value_counts().min() / labels_df['label'].value_counts().max():.2%}")
print(f"Output directory: {combined_dir}")
print(f"Labels file: {csv_path}")
print("=" * 60)