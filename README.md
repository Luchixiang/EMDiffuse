# EMDiffuse

This repository contains the official Pytorch implementation of our paper: **EMDiffuse: A Diffusion-based Deep Learning Method Augmenting Ultrastructural Imaging and Volume Electron Microscopy** accepted by [Nature Communications](https://www.nature.com/articles/s41467-024-49125-z).

EMDiffuse offers a toolkit for applying diffusion models to electron microscopy (EM) images, designed to enhance ultrastructural imaging in EM and extend the capabilities of volume electron microscopy (vEM). We have tailored the diffusion model for EM applications, developing **EMDiffuse-n** for EM denoising, **EMDiffuse-r** for EM super-resolution, and **vEMDiffuse-i** and **vEMDiffuse-a** for generating isotropic resolution data from anisotropic volumes in vEM.  

A selection of model weights is available at  [EMDiffuse_model_weight](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/EtSvqrIyrNREim5dJfabx2ABMLNhwk2Z9EsJDD4w6mls8g?e=OdP4Vq). Download them and place them in the `./experiments` folder.  The vEMDiffuse-i model was trained on the  [Openorgnelle liver dataset](https://doi.org/10.25378/janelia.16913047.v1). And vEMDiffuse-a was trained on the [MICrONS multi-area dataset](https://www.microns-explorer.org/). 

All results, including training and inference, will be stored in a newly created folder under `./experiments`. 

Running the diffusion process on a **GPU** is highly recommended for both training and testing.

For more information, please visit our webpage: https://www.haibojianglab.com/emdiffuse.

Should you have any questions regarding the code, please do not hesitate to contact us.

### Update:

1. Experiments on anisotropic volumes with fewer layers (e.g., 128 layers) are now possible. Given our input size 256, we duplicate the anisotropic volume along the axial axis before proceeding with vEMDiffuse-a training. (15/08/2023)

### Dependency

Please install PyTorch (=1.13) before you run the code. We strongly recommend you install Anaconda3, where we use Python 3.8.  

```
conda create --name emdiffuse python=3.8
conda activate emdiffuse
pip install -r requirements.txt
```

## Jupyter notebooks 

Explore our Jupyter notebooks for step-by-step tutorials:

- [2D EM denoising](https://github.com/Luchixiang/EMDiffuse/tree/master/example/denoise)
- [2D EM super-resolution](https://github.com/Luchixiang/EMDiffuse/tree/master/example/super-res)
- [Isotropic vEM reconstruction with isotropic training data](https://github.com/Luchixiang/EMDiffuse/tree/master/example/vEMDiffuse-i)
- [Isotropic vEM reconstruction without isotropic training data](https://github.com/Luchixiang/EMDiffuse/tree/master/example/vEMDiffuse-a)

In order to run the notebooks, install jupyter in your conda environment or use [Google Colab](https://colab.research.google.com/). 

```
pip install jupyter
```

## Instructions for EMDiffuse-n (2D EM denoising)

### Training:
#### Step 1: Download the Dataset
Download the dataset from https://zenodo.org/records/10205819.
#### Step 2:  Align and Crop
Register and crop patches for model training. For our dataset with multiple noise levels:

```python
cd RAFT/core
python register.py --path /data/EMDiffuse_dataset/brain_train --tissue Brain --patch_size 256 --overlap 0.125
```

Replace the `path` with your dataset's file path. `patch_size` should be a power of two or divisible by 8, and `overlap` sets the overlap ratio of adjacent patches. 

For transfer learning on other samples, replace ``tissue`` with the target, such as the `Liver`, `Heart`, or `BM`. 

For your own denoise dataset with file structure:

```
Denoise_dataset
	img
		1.tif
		2.tif
		3.tif
		...
	gt
		1.tif
		2.tif
		3.tif
		...
```

```
cd RAFT/core
python register_custom.py --path /data/EMDiffuse_dataset/brain_train --patch_size 256 --overlap 0.125
```

Replace the `path` with your dataset's file path.

#### Step 3: Model Training

```python
cd ../..
python run.py -c config/EMDiffuse-n.json -b 16 --gpu 0,1,2,3 --port 20022 --path /data/EMDiffuse_dataset/brain_train/denoise/train_wf --lr 5e-5
```

`gpu` denotes the GPU devices to be used during training. Multiple GPU training is supported. 

Both the model's state and its training metrics are automatically saved within a newly created directory, `./experiments/train_EMDiffuse-r_time`. Here, `time` is a placeholder for the actual timestamp when the training session begins. 

### Inference

#### Step 1: Download the Dataset

Download the dataset from https://zenodo.org/records/10205819.

#### Step 2: Crop Image

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task denoise
```

#### Step 3: Testing

Download the [model weight](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/EtSvqrIyrNREim5dJfabx2ABMLNhwk2Z9EsJDD4w6mls8g?e=OdP4Vq) and place them in the `./experiments` folder. 

```python
python run.py -p test -c config/EMDiffuse-n.json --gpu 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/denoise_test_crop_patches --resume ./experiments/EMDiffuse-n/best --mean 1 --step 1000
```

The diffusion model samples one plausible solution from the learned solution distribution. `mean` denotes the number of outputs to generate and averaging (each output and averaged output will be saved). `resume` indicates the path to the model's weight file. `step` controls the number of diffusion steps, with more steps generally leading to higher image quality.

## Instructions for EMDiffuse-r (2D EM super-resolution)

### Training:

#### Step 1: Download the Dataset

Download the dataset from https://zenodo.org/records/10205819.

#### Step 2: Align and Crop

```python
cd RAFT/core
python super_res_register.py --path /data/EMDiffuse_dataset/brain_train --patch_size 128 --overlap 0.125
```

#### Step 3: Model Training

```python
cd ../..
python run.py -c config/EMDiffuse-r.json -b 16 --gpu 0,1,2,3 --port 20022 --path /data/EMDiffuse_dataset/brain_train/zoom/train_wf --lr 5e-5
```

### Inference

#### Step 1: Download the Dataset

Download the dataset from https://zenodo.org/records/10205819.

#### Step 2 Crop Images

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task super
```

#### Step 3 Testing

Download the [model weight](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/EtSvqrIyrNREim5dJfabx2ABMLNhwk2Z9EsJDD4w6mls8g?e=OdP4Vq) and place them in the `./experiments` folder. 

```python
python run.py -p test -c config/EMDiffuse-r.json --gpu 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/super_test_crop_patches --resume ./experiments/EMDiffuse-r/best --mean 1 --step 1000
```

## Instructions for EMDiffuse-n and EMDiffuse-r Inference with Your Own EM Dataset

**These instructions are tailored for our denoising and super-resolution datasets. For your own dataset, you may need to adjust the cropping and registration codes to suit your data format. Here's a simple demonstration for performing inference on a dataset with the following file structure:**

```bash
test_images:
		image1.tif
		image2.tif
		....
```

#### Step1: Crop Image

```python
python crop_single_file.py --path ./test_images --task denoise 
```
#### Step 2: Testing

```python
python run.py -p test -c config/EMDiffuse-n.json --gpu 0 -b 60 --path ./test_images/denoise_test_crop_patches/ --resume ./experiments/EMDiffuse-n/best --mean 1 --step 1000
```

## Instructions for vEMDiffuse (Isotropic Reconstruction in vEM)

### Training

#### Data Preparation

Download or prepare your vEM training data. The training file structure should be like:

```
vEM_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The (n+1)th layer
```

#### Training vEMDiffuse-i with Isotropic Training Data
```
python run.py -c config/vEMDiffuse-i.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data -z 6 --lr 5e-5
```

``-z`` means the subsampling factor of the Z axis. For example, to reconstruct an 8 nm x 8 nm x 8 nm volume from an 8 nm x 8 nm x 48 nm volume, the subsampling factor should be 6. 

####  Training vEMDiffuse-a w/o Isotropic Training Data

Slice along YZ view:
 ```
 python vEMa_pre.py --path ./vEM_data
 ```

Training

```
python run.py -c config/vEMDiffuse-a.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data/transposed -z 6 --lr 5e-5
```

### Testing 

To test, prepare an anisotropic volume. Execute the model using the appropriate configuration and model weights for isotropic reconstruction.

```
vEM_test_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The (n+1)th layer
```

```
python run.py -p test -c config/vEMDiffuse-i.json --gpu 0 -b 16 --path ./vEM_test_data/ -z 6 --resume ./experiments/vEMDiffuse-i/best --mean 1 --step 200
```

Adjust the model weight directory to where your best model weights are saved. 
