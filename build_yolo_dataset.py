import os
import glob
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def clean_plant_name(name):
    return name.replace('_rawframes', '').replace('_raw frames', '')

def main():
    base_dir = '/home/cvl/farmos_env/TrainingSet_22ndapr'
    output_dir = '/home/cvl/farmos_env/YOLO_Plant_Dataset_22ndapr'
    
    source_subdirs = [
        'Segmented_PCS_21apr',
        'Segmented_Fruits_Buds_21apr',
        'Segmented_PCS_Single_Leaf_21apr'
    ]
    
    # Find all unique plant names
    raw_plants = set()
    for subdir in source_subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for plant_dir in os.listdir(subdir_path):
                if os.path.isdir(os.path.join(subdir_path, plant_dir)):
                    raw_plants.add(plant_dir)
                    
    print(f"Found {len(raw_plants)} raw plant classes.")
    
    target_count = 500
    train_ratio = 0.8
    train_count = int(target_count * train_ratio)
    
    # Clean the output directory first to prevent mixing with the old run
    if os.path.exists(output_dir):
        print(f"Clearing old dataset at {output_dir}...")
        shutil.rmtree(output_dir)
    
    for raw_plant in raw_plants:
        clean_name = clean_plant_name(raw_plant)
        selected_images = []
        
        wp_images = glob.glob(os.path.join(base_dir, 'Segmented_PCS_21apr', raw_plant, '*.*'))
        buds_images = glob.glob(os.path.join(base_dir, 'Segmented_Fruits_Buds_21apr', raw_plant, '*.*'))
        leaf_images = glob.glob(os.path.join(base_dir, 'Segmented_PCS_Single_Leaf_21apr', raw_plant, '*.*'))
        
        random.shuffle(wp_images)
        random.shuffle(buds_images)
        random.shuffle(leaf_images)
        
        # 1. Whole plant (target 330)
        taken_wp = wp_images[:330]
        deficit_wp = 330 - len(taken_wp)
        selected_images.extend([('Segmented_PCS_21apr', img) for img in taken_wp])
        
        # 2. Buds (target 120 + any deficit from whole plant)
        target_buds = 120 + deficit_wp
        taken_buds = buds_images[:target_buds]
        deficit_buds = target_buds - len(taken_buds)
        selected_images.extend([('Segmented_Fruits_Buds_21apr', img) for img in taken_buds])
        
        # 3. Leaf (target 50 + any deficit from buds)
        target_leaf = 50 + deficit_buds
        taken_leaf = leaf_images[:target_leaf]
        selected_images.extend([('Segmented_PCS_Single_Leaf_21apr', img) for img in taken_leaf])
        
        # 4. Final Fallback (if we still don't have 500, it means buds and leaf were both short, but WP might have extra!)
        if len(selected_images) < target_count:
            deficit_final = target_count - len(selected_images)
            unused_wp = wp_images[330:]
            taken_extra_wp = unused_wp[:deficit_final]
            selected_images.extend([('Segmented_PCS_21apr', img) for img in taken_extra_wp])
            
            deficit_final -= len(taken_extra_wp)
            if deficit_final > 0:
                unused_buds = buds_images[target_buds:]
                taken_extra_buds = unused_buds[:deficit_final]
                selected_images.extend([('Segmented_Fruits_Buds_21apr', img) for img in taken_extra_buds])
                
                deficit_final -= len(taken_extra_buds)
                if deficit_final > 0:
                    unused_leaf = leaf_images[target_leaf:]
                    taken_extra_leaf = unused_leaf[:deficit_final]
                    selected_images.extend([('Segmented_PCS_Single_Leaf_21apr', img) for img in taken_extra_leaf])
        
        # Shuffle the final selected images so train/val are randomly distributed
        random.shuffle(selected_images)
        
        # Split
        train_images = selected_images[:train_count]
        val_images = selected_images[train_count:]
        
        # Create directories
        train_dir = os.path.join(output_dir, 'train', clean_name)
        val_dir = os.path.join(output_dir, 'val', clean_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Copy files
        print(f"Copying {len(train_images)} train and {len(val_images)} val images for {clean_name}...")
        for subdir, img in train_images:
            new_filename = f"{subdir}_{os.path.basename(img)}"
            shutil.copy2(img, os.path.join(train_dir, new_filename))
        for subdir, img in val_images:
            new_filename = f"{subdir}_{os.path.basename(img)}"
            shutil.copy2(img, os.path.join(val_dir, new_filename))

    print("Dataset generation complete!")

if __name__ == '__main__':
    main()
