import os
import pandas as pd 
import nibabel as nib 
import numpy as np 
from fastai.vision.all import *
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from tqdm.notebook import tqdm
import csv

# Preprocessing inspired by #TWIMLfest: Fundamentals of Medical Image Processing for Deep Learning, Sam Charrington and Tenebris97 on Kaggle

#*******************************************************
# Stock data in dataframe
list = []
for directory, _, files in os.walk('./segmentations'):
    for file in files:
        list.append((directory, file))

for directory, _, files in os.walk('./volume'):
    for file in files:
        list.append((directory, file))

df = pd.DataFrame(list, columns = ['directory', 'file'])
df.sort_values(by = ['file'], ascending=True)

#********************************************************

# Pair up the volume scan with its segmentation mask
df["mask_directory"] = ""
df["mask_file"] = ""

for i in range(131):
    scan, mask = f"volume-{i}.nii", f"segmentation-{i}.nii"

    df.loc[df['file'] == scan, 'mask_file'] = mask
    df.loc[df['file'] == scan, 'mask_directory'] = "./segmentations"

# Erase the unmatched scans and duplicates
df = df[df.mask_file != ''].sort_values(by = ['file']).reset_index(drop = True)    

#*********************************************************

# Read nii files
def read_nii(file_path):
    """
    Load, extract and apply a 90 degrees rotation 
    (the raw data is flipped) to the NIfTI file 
    found at the filepath
    """
    scan = nib.load(file_path)
    data = np.rot90(np.array(scan.get_fdata()))
    return data

#********************************************************
# Classification functions

# Checks if the scan contains the liver or not
def is_empty(mask_image):
    return int (not (1 in mask_image))

# Checks if the liver is sane or not
def is_tumor(mask_image):
    return int((2 in mask_image))    
#*********************************************************
#TWIMLfest: Fundamentals of Medical Image Processing for Deep Learning, Sam Charrington

# Window the images so that the liver is more visible
def apply_windowing(image, level, width):
    windowed_image = image.copy()
    min = level - width // 2
    max = level + width // 2

    windowed_image[windowed_image < min] = min
    windowed_image[windowed_image > max] = max
    
    return windowed_image

#https://radiopaedia.org/articles/windowing-ct?lang=us
liver_level = 30
liver_width = 150
#*********************************************************

# Plotting some sample
idx = 0
nii_sample = read_nii(df.loc[idx, 'directory'] + "/" + df.loc[idx, 'file'])
sample_mask = read_nii(df.loc[idx, 'mask_directory'] + "/" + df.loc[idx, 'mask_file'])

scan_sample = nii_sample[..., 55].astype(np.float32)

scan_sample_windowed = apply_windowing(scan_sample, liver_level, liver_width)

# Create a 1x2 grid for plotting
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Plot the original image
ax0 = plt.subplot(gs[0])
ax0.imshow(scan_sample, cmap = plt.cm.bone)
ax0.set_title('Original Image')

# Plot the windowed image
ax1 = plt.subplot(gs[1])
ax1.imshow(scan_sample_windowed, cmap=plt.cm.bone)
ax1.set_title('Windowed Image')

plt.show()

#**********************************************************
#TWIMLfest: Fundamentals of Medical Image Processing for Deep Learning, Sam Charrington

# Window the images so that the liver is more visible

#https://radiopaedia.org/articles/windowing-ct?lang=us
dicom_windows = types.SimpleNamespace(liver=(150,30), custom = (200,60))

idx = 0
sample_scan = read_nii(df.loc[idx,'directory']+"/"+df.loc[idx,'file'])
sample_mask = read_nii(df.loc[idx,'mask_directory']+"/"+df.loc[idx,'mask_file'])

idx = 55
sample_slice = tensor(sample_scan[...,idx].astype(np.float32))

class TensorScan(TensorImageBW):

    def windowed(self: Tensor, w, l):
        px = self.clone()
        px_min = l - w // 2
        px_max = l + w // 2
        px[px < px_min] = px_min
        px[px > px_max] = px_max
        return (px - px_min) / (px_max - px_min)

    def freqhist_bins(self: Tensor, n_bins=100):
        imsd = self.view(-1).sort()[0]
        t = torch.cat([tensor([0.001]),
                       torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                       tensor([0.999])])
        t = (len(imsd) * t).long()
        return imsd[t].unique()

    def normalize(self: Tensor, brks=None):
        if brks is None:
            brks = self.freqhist_bins()
        ys = np.linspace(0., 1., len(brks))
        x = self.numpy().flatten()
        x = np.interp(x, brks.numpy(), ys)
        return torch.clamp(tensor(x).reshape(self.shape), 0., 1.)

    def to_n_channel(self: Tensor, wins, bins=None):
        res = [self.windowed(*win) for win in wins]
        if not isinstance(bins, int) or bins != 0:
            res.append(self.normalize(bins).clamp(0, 1))
        dim = [0, 1][self.dim() == 3]
        return TensorScan(torch.stack(res, dim=dim))

    def save_jpg(self: Tensor, path, wins, bins=None, quality=120):
        fn = Path(path).with_suffix('.jpg')
        x = (self.to_n_channel(wins, bins) * 255).byte()
        im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
        im.save(fn, quality=quality)

_,axs = subplots(1,1)

sample_slice = TensorScan(sample_slice)

sample_slice.save_jpg('test.jpg', [dicom_windows.liver, dicom_windows.custom])
show_image(Image.open('test.jpg'), ax=axs[0], figsize=(8, 6))
plt.show()


#***********************************************************************************
# Generate the database

generate_jpg = True
scan_info = []

if(generate_jpg):

    os.makedirs('scans', exist_ok=True)
    os.makedirs('masks', exist_ok=True)

    for i in tqdm(range(len(df))): 
        scan        = read_nii(df.loc[i,'directory'] + "/" + df.loc[i,'file'])
        mask        = read_nii(df.loc[i,'mask_directory'] + "/" + df.loc[i,'mask_file'])
        file_name   = str(df.loc[i,'file']).split('.')[0]
        dimension   = scan.shape[2] 

        for slice in range(dimension): 
            mask_img = mask[..., slice]
            scan_img = TensorScan(scan[..., slice].astype(np.float32))
            
            if (not is_empty(mask_img)): # keep only scan containing liver
                # Save scan information to the list
                scan_info.append({'scan_name': file_name,
                                  'is_tumor': is_tumor(mask_img),
                                  'file_path': f"scans/{file_name}_slice_{slice}.jpg",
                                  'mask_path': f"masks/{file_name}_slice_{slice}_mask.png"
                                })
                                
                # Save enhanced colored scan and grayscale mask
                #scan_img.save_jpg(f"scans/{file_name}_slice_{slice}.jpg", [dicom_windows.liver,dicom_windows.custom])
                #plt.imsave(f"masks/{file_name}_slice_{slice}_mask.png", mask_img, cmap='bone')


# Save scan information in a CSV file   
csv_filename = 'scan_info.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['scan_name', 'is_tumor', 'file_path', 'mask_path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for info in scan_info:
        writer.writerow(info)

print(f"Scan information saved to {csv_filename}")


#*******************************************************************************
# Dividing in training, testing and validation

import shutil
from sklearn.model_selection import train_test_split

df_data = pd.read_csv('scan_info.csv')

# Split the dataset into train, validation, and test sets
train_df, temp_df = train_test_split(df_data, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create directories for train, validation, and test sets
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for scans and masks within train, validation, and test sets
train_scans_dir = os.path.join(train_dir, 'train_scans')
train_masks_dir = os.path.join(train_dir, 'train_masks')
valid_scans_dir = os.path.join(valid_dir, 'valid_scans')
valid_masks_dir = os.path.join(valid_dir, 'valid_masks')
test_scans_dir = os.path.join(test_dir, 'test_scans')
test_masks_dir = os.path.join(test_dir, 'test_masks')

os.makedirs(train_scans_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(valid_scans_dir, exist_ok=True)
os.makedirs(valid_masks_dir, exist_ok=True)
os.makedirs(test_scans_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# Move scans and masks to their respective directories
def move_images(df, scans_dir, masks_dir):
    for _, row in df.iterrows():
        shutil.move(row['file_path'], os.path.join(scans_dir, os.path.basename(row['file_path'])))
        shutil.move(row['mask_path'], os.path.join(masks_dir, os.path.basename(row['mask_path'])))

move_images(train_df, train_scans_dir, train_masks_dir)
move_images(valid_df, valid_scans_dir, valid_masks_dir)
move_images(test_df, test_scans_dir, test_masks_dir)


print("Scans and masks split into train, validation, and test sets.")

# Add a new 'division' column to the DataFrame with conditions
df_data['division'] = df_data.apply(lambda row: 'train' if row['file_path'] in train_df['file_path'].values
                                       else ('valid' if row['file_path'] in valid_df['file_path'].values
                                             else ('test' if row['file_path'] in test_df['file_path'].values
                                                   else '')), axis=1)

csv_filename = 'scan_info.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['scan_name', 'is_tumor', 'file_path', 'mask_path', 'division']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for _, data in df_data.iterrows():
        writer.writerow({
            'scan_name': data['scan_name'],
            'is_tumor': data['is_tumor'],
            'file_path': data['file_path'],
            'mask_path': data['mask_path'],
            'division': data['division']
        })

