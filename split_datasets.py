import os
import shutil
import random


original_dataset_path = 'datasets256'     # Original dataset path
new_dataset_path = 'datasets'                       # Divide the dataset path

train_path = os.path.join(new_dataset_path, 'train')
val_path = os.path.join(new_dataset_path, 'val')
test_path = os.path.join(new_dataset_path, 'test')

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

classes = os.listdir(original_dataset_path)     # Get all categories

random.seed(42)

for class_name in classes:      # Traverse each category
    
    src_folder = os.path.join(original_dataset_path, class_name)    # Source folder path
    
    # Check if the folder for this category already exists under train, val, and test
    train_folder = os.path.join(train_path, class_name)
    val_folder = os.path.join(val_path, class_name)
    test_folder = os.path.join(test_path, class_name)

    # If the train, val, and test folders already exist, skip the folder creation section
    if os.path.exists(train_folder) and os.path.exists(val_folder) and os.path.exists(test_folder):
        # Check if the folder is empty
        if os.listdir(train_folder) and os.listdir(val_folder) and os.listdir(test_folder):
            print(f"Category {class_name} already exists and is not empty, skip processing.")
            continue

    # create folder
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    
    files = os.listdir(src_folder)      # Retrieve all file names under this category
    files = files[:10000]       # Only retrieve the first 10000 files
    random.shuffle(files)       # Shuffle file list

    total_files = len(files)
    train_split_index = int(total_files * 0.8)
    val_split_index = int(total_files * 0.9)

    train_files = files[:train_split_index]
    val_files = files[train_split_index:val_split_index]
    test_files = files[val_split_index:]

    for file in train_files:
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(train_folder, file)
        shutil.copy(src_file, dst_file)

    for file in val_files:
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(val_folder, file)
        shutil.copy(src_file, dst_file)

    for file in test_files:
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(test_folder, file)
        shutil.copy(src_file, dst_file)

print("Dataset partitioning completed!")