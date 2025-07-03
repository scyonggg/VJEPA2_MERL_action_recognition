#!/usr/bin/env python3
"""
Convert MERL dataset from JSON format to CSV format for V-JEPA2 temporal action localization.
"""

import json
import os
import pandas as pd
from pathlib import Path

def convert_merl_to_csv(json_path, output_dir):
    """
    Convert MERL dataset from JSON to CSV format for temporal action localization.
    
    Args:
        json_path: Path to merl_data.json
        output_dir: Directory to save CSV files
    """
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train, validation, and test splits
    for split in ['train', 'validation', 'test']:
        if split not in data:
            print(f"Warning: {split} split not found in JSON data")
            continue
            
        rows = []
        for item in data[split]:
            video_path = item['video_path']
            
            # Update video path to use the correct location
            # Extract filename from the path
            filename = os.path.basename(video_path)
            # Update to use the correct MERL dataset location
            new_video_path = f"/mnt/data/datasets/MERL/Videos_MERL_Shopping_Dataset/{filename}"
            
            # For each annotation, create a row
            for annotation in item['annotations']:
                label = annotation['label']
                start_frame, end_frame = annotation['segment']
                
                # Convert label to class index
                label_to_idx = {
                    'Reach To Shelf': 0,
                    'Retract From Shelf': 1,
                    'Hand In Shelf': 2,
                    'Inspect Product': 3,
                    'Inspect Shelf': 4
                }
                
                class_idx = label_to_idx.get(label, -1)
                if class_idx == -1:
                    print(f"Warning: Unknown label {label}")
                    continue
                
                # Format: video_path, start_frame, end_frame, label, label_name
                rows.append({
                    'video_path': new_video_path,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'label': class_idx,
                    'label_name': label
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        # Map validation to val for CSV filename
        csv_filename = 'val_paths.csv' if split == 'validation' else f'{split}_paths.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False, header=False, sep=' ')
        
        print(f"Created {csv_path} with {len(df)} samples")
        print(f"Class distribution:")
        print(df['label_name'].value_counts())
        print(f"Temporal segment statistics:")
        print(f"  Average segment length: {(df['end_frame'] - df['start_frame']).mean():.1f} frames")
        print(f"  Min segment length: {(df['end_frame'] - df['start_frame']).min()} frames")
        print(f"  Max segment length: {(df['end_frame'] - df['start_frame']).max()} frames")

def main():
    # Paths
    json_path = "MERL/merl_data.json"
    output_dir = "MERL/csv_format"
    
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return
    
    # Convert to CSV
    convert_merl_to_csv(json_path, output_dir)
    print("Conversion completed!")

if __name__ == "__main__":
    main() 