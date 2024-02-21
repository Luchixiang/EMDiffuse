# EMDiffuse

This repository is the official Pytorch implementation of our paper: **EMDiffuse: a diffusion-based deep learning method augmenting ultrastructural imaging and volume electron microscopy**.

EMDiffuse is a package for the application of diffusion models on electron microscopy images, aiming to enhance EM ultrastructural imaging and expand the realm of vEM capabilities. Here, we adopted the diffusion model for EM applications and developed **EMDiffuse-n** for EM denoising, **EMDiffuse-r** for EM super-resolution, and **vEMDiffuse-i** and **vEMDiffuse-a** for generating isotropic resolution data from anisotropic volumes for vEM. All the results, including the training and inference in the following instructions, will be saved in a newly created folder in `./experiments`. Also, you should replace ``--path`` arguments in all instructions into your dataset path.

Several representative model weights have been uploaded to [EMDiffuse_model_weight](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/EtSvqrIyrNREim5dJfabx2ABMLNhwk2Z9EsJDD4w6mls8g?e=OdP4Vq). You can download them and place them in the ``./experiments`` folder.  The weight in vEMDiffuse-i was trained on the [Openorgnelle liver dataset](https://doi.org/10.25378/janelia.16913047.v1). The weight in vEMDiffuse-a was trained on the [MICrONS multi-area dataset](https://www.microns-explorer.org/). 

You can also visit our webpage for more information: https://www.haibojianglab.com/emdiffuse. 

It is highly recommended that the diffusion process be run on **GPU** both of training and testing.

Please feel free to contact us if you have any questions about the code.

### Update:

1. Experiment on anisotropic volume with fewer layers(e.g., 128 layers). Since our input size is 256, we duplicate the anisotropic volume along the axial axis and then do vEMDiffuse-a training. (15/08/2023)

### Dependency

Please install PyTorch (=1.13) before you run the code. We strongly recommend you install Anaconda3, where we use Python 3.8.  

```
conda create --name emdiffuse python=3.8
conda activate emdiffuse
pip install -r requirements.txt
```

## Jupyter notebooks 

Have a look at our jupyter notebook:

- [2D EM denoising](https://github.com/Luchixiang/EMDiffuse/tree/master/example/denoise)
- [2D EM super-resolution](https://github.com/Luchixiang/EMDiffuse/tree/master/example/super-res)
- [Isotropic vEM reconstruction with isotropic training data](https://github.com/Luchixiang/EMDiffuse/tree/master/example/vEMDiffuse-i)
- [Isotropic vEM reconstruction without isotropic training data](https://github.com/Luchixiang/EMDiffuse/tree/master/example/vEMDiffuse-a)

In order to run the notebooks, install jupyter in your conda environment or try [Google Colab](https://colab.research.google.com/). 

```
pip install jupyter
```

## EMDiffuse-n for 2D EM denoising

### Training:
#### Step 1: Download the Dataset
Download the dataset from https://zenodo.org/records/10205819
#### Step 2:  Align and Crop
This step automatically registers and crops the patch for model training.  For our multiple noise level dataset

```python
cd RAFT/core
python register.py --path /data/EMDiffuse_dataset/brain_train --tissue Brain --patch_size 256 --overlap 0.125
```

Replace the argument  `--path` with the file path on which you place the dataset.  `patch_size` denotes the image dimensions of each cropped patch. As a general rule, use a patch size that is a power of two along all axes, or which is at least divisible by 8.  `overlap` is the overlap ratio of adjacent patches. 

For transfer learning on other samples, please replace ``--tissue`` with the target such as the `Liver`, `Heart`, or `BM`. 

For your own denoise dataset with a structure like:

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

Replace the argument  `--path` with the file path on which you place the dataset. 

#### Step 3: Model Training

```python
cd ../..
python run.py -c config/EMDiffuse-n.json -b 16 --gpu 0,1,2,3 --port 20022 --path /data/EMDiffuse_dataset/brain_train/denoise/train_wf --lr 5e-5
```

`gpu` denotes the GPU devices to be used during training. Multiple GPU training is supported. 

Both the model's state and its training metrics are automatically saved within a newly created directory, `./experiments/train_EMDiffuse-r_time` as logged above. Here, `time` is a placeholder for the actual timestamp when the training session begins, ensuring each training session is uniquely identifiable. Furthermore, you can look at the predictions for some validation images, which can help recognize problems early on.

### Inference

#### Step 1: Download the Dataset

Download the dataset from https://zenodo.org/records/10205819

#### Step 2: Crop Image

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task denoise
```

#### Step 3: Testing

```python
python run.py -p test -c config/EMDiffuse-n.json --gpu 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/denoise_test_crop_patches --resume ./experiments/EMDiffuse-n/best --mean 1 --step 1000
```

The diffusion model samples one plausible solution from the learned solution distribution. `mean` denotes the number of outputs you want to generate and averaging.  `resume` denotes the path to the model's weights file and the epoch number from which to load the model. `step` is the number of diffusion steps. More steps, higher image quality. 

## EMDiffuse-r for EM super-resolution

### Training:

#### Step 1: Download the Dataset

Download the dataset from https://zenodo.org/records/10205819

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

Download the dataset from https://zenodo.org/records/10205819

#### Step 2 Crop Images

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task super
```

#### Step 3 Testing

```python
python run.py -p test -c config/EMDiffuse-r.json --gpu 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/super_test_crop_patches --resume ./experiments/EMDiffuse-r/best --mean 1 --step 1000
```

## EMDiffuse-n and EMDiffuse-r Inference for Your Own EM Dataset

**All the above steps are designed for our multiple noise level denoising and super-resolution datasets. If you want to try your own dataset, you may need to modify the crop and register codes to meet the format of required dataset. We provide a simple demo here for you to implement inference on a dataset with the file structure:**

```bash
test_images:
		image1.tif
		image2.tif
		....
```

i.e., all the noisy images or low-resolution images are in the same folder.

#### Step1: Crop Image

```python
python crop_single_file.py --path ./test_images --task denoise 
```
#### Step 2: Testing

```python
python run.py -p test -c config/EMDiffuse-n.json --gpu 0 -b 60 --path ./test_images/denoise_test_crop_patches/ --resume ./experiments/EMDiffuse-n/best --mean 1 --step 1000
```

## vEMDiffuse for vEM isotropic reconstruction

### Training

#### Data Preparation

Firstly, you should download volume data from [Openorganelle](https://openorganelle.janelia.org/) or use your own vEM data. The data structure should be like: 

```
vEM_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The (n+1)th layer
```

#### Training vEMDiffuse-i
For vEMDiffuse-i, the training data must be isotropic. 
```
python run.py -c config/vEMDiffuse-i.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data -z 6 --lr 5e-5
```

``-z`` means the subsampling factor of the Z axis. In this example, to reconstruct an 8 nm x 8 nm x 8 nm volume from an 8 nm x 8 nm x 48 nm volume, the subsampling factor should be 6. 

####  Training vEMDiffuse-a

For vEMDiffuse-a, we don't need the isotropic training data. The vEMa_pre.py first slices your volume along the lateral (Y axis) to construct the training data. The model learns the lateral information and applies it to improve axial resolution. 
 ```
 python vEMa_pre.py --path ./vEM_data
 python run.py -c config/vEMDiffuse-a.json -b 16 --gpu 0,1,2,3 --port 20022 --path ./vEM_data/transposed -z 6 --lr 5e-5
 ```

### Testing 

Firstly, you need to have an anisotropic volume with the data structure like: 

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

Don't forget to modify the model weight directory. 
