import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate, misc
import matplotlib
matplotlib.use('Agg')

input_dir = 'kaggle/train_simplified'
output_base_dir = 'datasets256'

os.makedirs(output_base_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]    # Retrieve all CSV files from the folder

skipped_files = []  # Record skipped files

for csv_file in csv_files:
    
    csv_file_path = os.path.join(input_dir, csv_file)   # Build a complete file path
    output_dir = os.path.join(output_base_dir, os.path.splitext(csv_file)[0])   # Build output directory

    if os.path.exists(output_dir):      # Check if the output directory exists
        skipped_files.append(csv_file)
        print(f'The directory already exists, skip file: {csv_file}')
        continue

    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(csv_file_path)       # Read CSV file
    
    for index, row in data.iterrows():  # Traverse each row of data
        drawing = eval(row['drawing'])
        key_id = row['key_id']
        word = row['word']
        
        img = np.zeros((256, 256))      # Initialize image
        fig = plt.figure(figsize=(256/96, 256/96), dpi=96)
        
        for stroke in drawing:      # Draw each stroke
            stroke_x = stroke[0]
            stroke_y = stroke[1]
            x = np.array(stroke_x)
            y = np.array(stroke_y)
            np.interp((x + y) / 2, x, y)
            plt.plot(x, y, 'k')
        
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{word}-{key_id}.png'))
        plt.close(fig)
        print(f'Conversion completed: {csv_file} the {index:06d}image')
        
print("The skipped files are:")
for file in skipped_files:
    print(file)