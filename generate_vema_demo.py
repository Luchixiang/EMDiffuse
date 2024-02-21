import os
from tifffile import imread, imwrite
import numpy as np
path = '/home/u3590540/cxlu/allen_brain_train_nopercent//0'
volume = []
for i in range(256):
    volume.append(imread(os.path.join(path, str(i) + '.tif')))
volume = np.stack(volume, axis=0)
for i in range(24):
    img = volume[:, i, :]
    imwrite(( './vEMDiffuse-a-demo/'+ str(i) + '.tif'), img)
