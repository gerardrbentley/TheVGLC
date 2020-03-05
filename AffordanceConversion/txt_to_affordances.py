import argparse
import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

AFFORDANCES = ["changeable", "dangerous", "destroyable", "gettable", "movable", 
        "portal", "solid", "ui", "usable"]

def main(args):
    """
    Converts VGLC processed levels from txt format to 10 channel, pixel-wise affordance maps.
    Requires a json mapping from encoded txt characters to affordances that maybe (0.5) or definitely (1.0) hold for that tile
    Fills in 0.0 for affordances not in mapping
    
    Basic usage from root vglc folder: 
        `python AffordancesConversion/txt_to_affordances.py --file ./Super\ Mario\ Bros/Processed/mario-1-1.txt
            --json ./AffordanceConversion/smb_affordances.json
            --verbose --visualize --no-save`
    """
    if args.folder != '':
        file_paths = glob.glob(os.path.join(args.folder, '*.txt'))
    else:
        file_paths = [args.file]
    try:
        with open(args.json) as f:
            affordance_dict = json.load(f)
            affordance_dict = affordance_dict['tiles']
    except:
        print('Error loading affordance json')
        return

    grid_size = args.grid_size
    for file_path in file_paths:
        if args.verbose:
            print(f'file: {file_path}')
            print(args)
            print('getting level rows and cols from file')
        file_name = os.path.split(file_path)[1]
        basename = os.path.splitext(file_name)[0]
        
        try:
            with open(file_path) as processed_level:
                level_rows = [[char for char in line if char != '\n'] for line in processed_level]
        except:
            print("Error loading level map")
            return
        num_cols = len(level_rows[0])
        num_rows = len(level_rows)
        for line in level_rows:
            if num_cols != len(line):
                print(f"Not all lines the same width: {num_cols}, {len(line)}")
                return

        height = num_rows * grid_size
        width = num_cols * grid_size
        affordance_map = np.zeros((height, width, len(AFFORDANCES)), dtype=np.float)

        buckets = np.zeros(len(AFFORDANCES))
        seen_chars = []
        for (row, col) in [(r, c) for r in range(num_rows) for c in range(num_cols)]:
            row_idx, col_idx = row*grid_size, col*grid_size
            curr_char = level_rows[row][col]
            char_affords = affordance_dict.get(curr_char, {})
            for affordance, value in char_affords.items():
                affordance_map[row_idx:row_idx+grid_size, col_idx:col_idx+grid_size, AFFORDANCES.index(affordance)] = value
                buckets[AFFORDANCES.index(affordance)] += int(grid_size*grid_size)
            if curr_char not in seen_chars and args.verbose:
                seen_chars.append(curr_char)
                print(f"Char {curr_char} Affords: {char_affords}")
                
        if args.verbose:
            for affordance in AFFORDANCES:
                print(f"Unique vals {affordance}: {np.unique(affordance_map[:,:,AFFORDANCES.index(affordance)])}")
            print(f"Num pixels per affordance: {buckets.tolist()}")
            print(AFFORDANCES)
            print(f"Total pixels {num_rows} rows, {num_cols} cols: (h*w) ({height} * {width}): {height*width}")
            print(f'Affordance map shape (h,w,channels): {affordance_map.shape}')
        if not args.no_save:
            os.makedirs(args.dest, exist_ok=True)
            dest_file = os.path.join(args.dest, f"{basename}.npy")
            print(f'saving to {dest_file}')
            np.save(dest_file, affordance_map)
    if args.visualize:
        print('visualizing last affordance map solid and permeable (--visualizeall for all affordances)')
        visualize(affordance_map, basename, only_solid_permeable=True)

def visualize(affordance_map, filename, only_solid_permeable=False):
    cmap = ListedColormap(['black', 'gray', 'white'])
    # Map 0 to black, 0.5 to gray (because 5 is between 4 and 6.. matplotlib), 1.0 to white
    affordance_map = np.copy(affordance_map)*10
    bounds = [0, 4, 6, 10]
    norm = BoundaryNorm(bounds, cmap.N)
    if not only_solid_permeable:
        fig, axeslist = plt.subplots(ncols=2, nrows=5)
        fig.suptitle(filename)
        for i in range(affordance_map.shape[2]):
            title = AFFORDANCES[i]
            axeslist.ravel()[i].imshow(affordance_map[:,:,i], cmap=cmap, norm=norm, interpolation='nearest')
            axeslist.ravel()[i].set_title(title)
            axeslist.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure('solid')
        plt.imshow(affordance_map[:,:,AFFORDANCES.index('solid')], cmap=cmap, norm=norm, interpolation='nearest')
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Textrue match detection')

    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--json', type=str, default="./AffordanceConversion/smb_affordances.json")
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--dest', type=str, default="./AffordanceConversion/SMB/")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualizeall', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)