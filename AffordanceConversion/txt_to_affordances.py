import argparse
import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

AFFORDANCES = ["changeable", "dangerous", "destroyable", "gettable", "movable", 
        "permeable", "portal", "solid", "ui", "usable"]

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

    grid_size = args.grid_size
    for file_name in file_paths:
        if args.verbose:
            print(f'file: {file_name}')
            print(args)
            print('getting level rows and cols from file')
        file_name = os.path.split(file_name)[1]
        basename = os.path.splitext(file_name)[0]
        
        try:
            with open(args.json) as f:
                affordance_dict = json.load(f)
                affordance_dict = affordance_dict['tiles']
            with open(args.file) as processed_level:
                level_rows = [[char for char in line if char != '\n'] for line in processed_level]
        except:
            print("Error loading affordance file or level map")
            return
        num_cols = len(level_rows[0])
        num_rows = len(level_rows)
        for line in level_rows:
            if num_cols != len(line):
                print(f"Not all lines the same width: {num_cols}, {len(line)}")
                return
        if args.verbose:
            print(f"Num rows: {num_rows}, num cols: {num_cols}")
            print(json.dumps(affordance_dict))

        height = num_rows * grid_size
        width = num_cols * grid_size
        affordance_map = np.zeros((height, width, len(AFFORDANCES)), dtype=np.float)

        
        seen_chars = []
        for (row, col) in [(r, c) for r in range(num_rows) for c in range(num_cols)]:
            row_idx, col_idx = row*grid_size, col*grid_size
            curr_char = level_rows[row][col]
            char_affords = affordance_dict.get(curr_char, {})
            for affordance, value in char_affords.items():
                affordance_map[row_idx:row_idx+grid_size, col_idx:col_idx+grid_size, AFFORDANCES.index(affordance)] = value

            if curr_char not in seen_chars and args.verbose:
                seen_chars.append(curr_char)
                print(f"Char {curr_char} Affords: {char_affords}")
                
        if args.verbose:
            for affordance in AFFORDANCES:
                print(f"Unique vals in affordance map for {affordance}: {np.unique(affordance_map[:,:,AFFORDANCES.index(affordance)])}")
        if args.visualize:
            visualize(affordance_map)
        if not args.no_save:
            dest_file = os.path.join(args.dest, f"{basename}.npy")
            print(f'saving to {dest_file}')
            np.save(dest_file, affordance_map)

def visualize(affordance_map):
    cmap = ListedColormap(['black', 'gray', 'white'])
    # Map 0 to black, 0.5 to gray (because 5 is between 4 and 6.. matplotlib), 1.0 to white
    affordance_map = np.copy(affordance_map)*10
    bounds = [0, 4, 6, 10]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axeslist = plt.subplots(ncols=2, nrows=5)
    for i in range(affordance_map.shape[2]):
        title = AFFORDANCES[i]
        axeslist.ravel()[i].imshow(affordance_map[:,:,i], cmap=cmap, norm=norm, interpolation='nearest')
        axeslist.ravel()[i].set_title(title)
        axeslist.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
            

def parse_args():
    parser = argparse.ArgumentParser(description='Textrue match detection')

    parser.add_argument('--file', type=str, default="./Super Mario Bros/Processed/mario-8-1.txt")
    parser.add_argument('--json', type=str, default="./AffordanceConversion/smb_affordances.json")
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--dest', type=str, default="./AffordanceConversion/SMB/")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--grid-size', type=int,
                        default=16, help='grid square size')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)