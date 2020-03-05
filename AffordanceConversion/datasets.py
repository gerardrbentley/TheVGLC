from torch import Tensor
from PIL import Image
import glob
import os
from collections import namedtuple
import torch
import numpy as np
import segmentation_transforms as TT

# Expects directory of .png's as root
# Transforms should include ToTensor (also probably Normalize)
# Can apply different transform to output, returns image as input and label

Sample = namedtuple('Sample', ['image', 'target'])

class GameLevelsDataset(torch.utils.data.Dataset):
    def __init__(self, root='./MM', transform=TT.ToTensor()):
        self.image_dir = os.path.join(root)
        # Get abs file paths
        img_filetype =  'bmp'
        self.image_list = glob.glob(f'{self.image_dir}/*.bmp')
        if len(self.image_list) == 0:
            img_filetype = 'png'
            print('No BMP images found, trying PNG')
            self.image_list = glob.glob(f'{self.image_dir}/*.png')
        
        self.data = []
        TARGET_SIZE = 224
        for i, filename in enumerate(self.image_list):
            image = Image.open(filename).convert('RGB')
            segmentation_filename = filename.replace(img_filetype, 'npy')
            affordance_map = np.load(segmentation_filename)
            print(image.size, affordance_map.shape)
            width, height = image.size
            rows = int(height // TARGET_SIZE)
            cols = int(width // TARGET_SIZE)
            ctr = 0
            for (r,c) in [(r*TARGET_SIZE,c*TARGET_SIZE) for r in range(rows) for c in range(cols)]:
                small_image = image.crop(box=(c, r, c + TARGET_SIZE, r + TARGET_SIZE))
                small_extrema = small_image.getextrema()
                if small_image.getbbox():
                    small_map = np.copy(affordance_map)[r:r+TARGET_SIZE, c:c+TARGET_SIZE, :]
                    self.data.append((small_image, small_map))
                    if small_image.size != (224,224):
                        print(f'SMaLL IMAGE wRONG SIZE: {small_image.size}')
                    if small_map.shape != (224,224,9):
                        print(f"SMAL MAP WRONG SHAPE: {small_map.shape}")
                    # small_image.save(f"./nonblack/{r}_{c}_{i}.png")
                else:
                    ctr += 1
                    # small_image.save(f'./allblack/{r}_{c}_{i}.png')
            print(f'{ctr} all black sections out of {rows * cols} = {rows} * {cols}. File: {filename}')
        # self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.data)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image, target = self.data[idx]

        if self.transform:
            image, target = self.transform(image, target)

        # sample = {'image': image, 'target': target}
        return Sample(image, target)

class GameImagesDataset(torch.utils.data.Dataset):
    def __init__(self, root='/faim/datasets/per_game_screenshots/super_mario_bros_3_usa', transform=TT.ToTensor()):
        self.image_dir = os.path.join(root)
        # Get abs file paths
        self.image_list = glob.glob(f'{self.image_dir}/*.png')

    
        # self.image_folders = next(os.walk(self.image_dir))[1]
        self.length = len(self.image_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        screenshot_file = self.image_list[idx]

        image = Image.open(screenshot_file).convert('RGB')
        target = image.copy()

        if self.transform:
            image, target = self.transform(image, target)

        # sample = {'image': image, 'target': target}
        return Sample(image, target)


def get_dataset(name, transform):
    paths = {
        "mm": ('./MM/', GameLevelsDataset),
        "test": ('./TST/', GameLevelsDataset),
    }
    root_path, dataset_function = paths[name]

    ds = dataset_function(root=root_path, transform=transform)
    return ds

def get_stats(name):
    datasets = {
        "smb": ([0.3711, 0.3652, 0.5469],[0.2973, 0.2772, 0.4554]),
        "mm": ([0.19928, 0.3196, 0.3268], [0.34876, 0.3755, 0.4020])
    }
    try: 
        return datasets[name]
    except:
        return datasets['mm']

# From pytorch thread: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/3
# NOTE: requires images to be of same size
def dataset_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)
    mean = 0.0
    var = 0.0
    k = 1
    test_image = dataset[0].image
    c,h,w = test_image.shape
    # print(f"c: {c}, h: {h}, w: {w}")
    # lin_image = test_image.view(test_image.size(0), -1)
    # print(lin_image.shape)
    # print(f"test mean: {torch.mean(lin_image, 1)}")
    # print(f"test std: {torch.std(lin_image, 1)}")
    for i_batch, sample_batched in enumerate(dataloader):
        # if i_batch % 5 == 0:
        #     print(f"tick: {i_batch}")
        num_samples = sample_batched.image.size(0)
        images = sample_batched.image.view(num_samples, sample_batched.image.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataset)
    # Calculate full mean before variance
    for i_batch, sample_batched in enumerate(dataloader):
        # if i_batch % 5 == 0:
        #     print(f"tick: {i_batch}")
        num_samples = sample_batched.image.size(0)
        images = sample_batched.image.view(num_samples, sample_batched.image.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    
    std = torch.sqrt(var / (len(dataset)*h*w))
    return mean.tolist(), std.tolist()

if __name__ == "__main__":

    print('test Game Level Dataset')    
    trainset = get_dataset("mm", transform=None)
    print(type(trainset))
    print(f'len trainset: {len(trainset)}')
    data = trainset[1]
    # data['image'].show()
    image = data.image
    target = data.target

    print(f'Image and Target with Transform = None')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.size)}, {(target.shape)}')
    print(f'extrema: [{image.getextrema()}], [{target.min()}, {target.max()}]')

    # CHECK MEAN AND STD
    tensorset = get_dataset("mm", transform=TT.ToTensor())
    test_image = tensorset[0].image
    c,h,w = test_image.shape
    # print(f"c: {c}, h: {h}, w: {w}")
    lin_image = test_image.view(test_image.size(0), -1)
    #print(lin_image.shape)
    print(f"test mean: {torch.mean(lin_image, 1)}")
    print(f"test std: {torch.std(lin_image, 1)}")
    mean, std = dataset_mean_std(tensorset)
    print(f"mm dataset mean: {mean}, std: {std}")
    
    # smb = get_dataset("smb", transform=TT.ToTensor())
    # smb_mean, smb_std = dataset_mean_std(smb)
    # print(f"smb dataset smb_mean: {smb_mean}, smb_std: {smb_std}")


    image_tensor, target_tensor = TT.ToTensor()(image, target)

    print(f'Image and Target with Transform = ToTensor')
    print(f'types: {type(image_tensor)}, {type(target_tensor)}')
    print(f'shapes: {(image_tensor.shape)}, {(target_tensor.shape)}')
    # print(f'extrema: [{image_tensor.getextrema()}], [{target_tensor.getextrema()}]')

    do_transforms = TT.get_transform(False)
    image, target = do_transforms(image, target)

    image, target = image.unsqueeze(0), target.unsqueeze(0)

    print(f'Image and Target post Batching')
    print(f'shapes: {(image.shape)}, {(target.shape)}')

    image = image.detach().cpu()
    target = target.detach().cpu()

    print(f'Image and Target post detach, cpu for viz')
    print(f'types: {type(image)}, {type(target)}')
    print(f'shapes: {(image.shape)}, {(target.shape)}')
    print(
        f'ranges: [{image.min()} - {image.max()}], [{target.min()} - {target.max()}]')

    # visualize_outputs(image, titles=['Image'])
