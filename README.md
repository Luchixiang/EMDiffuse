# EMDiffuse

This repository is the official Pytorch implementation of our paper: **EMDiffuse: a diffusion-based deep learning method augmenting ultrastructural imaging and volume electron microscopy**.

EMDiffuse is a package for the application of diffusion models on electron microscopy images, aiming to enhance EM ultrastructural imaging and expand the realm of vEM capabilities. Here, we adopted the diffusion model for EM applications and developed **EMDiffuse-n** for EM denoising, **EMDiffuse-r** for EM super-resolution, and **vEMDiffuse-i** and **vEMDiffuse-a** for generating isotropic resolution data from anisotropic volumes for vEM. All the results including the training and inference in the following instructions will be saved in ``./experiments`` folder. Also, you should replace ``--path`` arguments in all instructions into your dataset path.

Several representative model weight has been uploaded in: [EMDiffuse_model_weight](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/EtSvqrIyrNREim5dJfabx2ABMLNhwk2Z9EsJDD4w6mls8g?e=OdP4Vq) and place the folder in ``./experiments`` folder.  The weight in vEMDiffuse-i was trained on [Openorgnelle liver dataset](https://doi.org/10.25378/janelia.16913047.v1). The weight in vEMDiffuse-a was trained on [MICrONS multi-area dataset](https://www.microns-explorer.org/). 

You can also visit our webpage for more information: https://www.haibojianglab.com/emdiffuse. 

Please feel free to contact us if you have any questions about the code.

### Update:

1. find some small misalginment between the published version and our local version in 'dataset' code line 221: replace // with %.  (12/08)



### Dependency

Please install PyTorch (=1.13) before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.8.  

```
conda create --name emdiffuse python=3.8
conda activate emdiffuse
pip install -r requirements.txt
```



## EMDiffuse-n for EM denoising

### Training:
#### Step 1
Download the dataset from https://zenodo.org/record/8136295
#### Step 2
Register the image. This step automatically registers and crops the patch for model training. 

```python
cd RAFT/core
python register.py --path /data/EMDiffuse_dataset/brain_train --tissue brain
```

You should replace the argument  `--path` with the file path you place the dataset. 

For transfer learning on other tissue, please replace ``--tissue`` with the target tissue such as the liver, heart, or bone marrow. 

#### Step 3

Model Training. Please change the file path in line 23 of the config file. 

```python
python run.py -c config/EMDiffuse-n.json -b 16 -gpu 0,1,2,3 --port 20022 --path /data/EMDiffuse_dataset/brain_train/denoise/train_wf
```

The model weight will be saved in experiments.

### Inference

#### Step 1

Download the dataset from https://zenodo.org/record/8136295

#### Step 2

Crop the image

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task denoise
```

#### Step 3

```python
python run.py -p test -c config/EMDiffuse-n.json -g 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/denoise_test_crop_patches
```



## EMDiffuse-r for EM super-resolution

### Training:

#### Step 1

Download the dataset from https://zenodo.org/record/8136295

#### Step 2

Register the image

```python
cd RAFT/core
python register-super-res.py --path /data/EMDiffuse_dataset/brain_train
```

#### Step 3

Model Training

```python
python run.py -c config/EMDiffuse-n.json -b 16 -gpu 0,1,2,3 --port 20022 --path /data/EMDiffuse_dataset/brain_train/zoom/train_wf
```

### Inference

#### Step 1

Download the dataset from https://zenodo.org/record/8136295

#### Step 2

Crop the image

```python
python test_pre.py --path /data/EMDiffuse_dataset/brain_test --task super
```

#### Step 3

```python
python run.py -p test -c config/EMDiffuse-n.json -g 0 -b 60 --path /data/EMDiffuse_dataset/brain_test/super_test_crop_patches
```

## EMDiffuse-n and EMDiffuse-r For your own EM dataset

**Note that all the above steps are designed for our multiple noise level denoising and super-resolution datasets. If you want to try your own dataset, you may need to modify the crop and register codes to meet the format of your dataset. We provide a demo for you to implement inference on a simpler dataset with the file structure:**

```bash
Test_images:
		image1.tif
		image2.tif
		....
```

i.e., all the noisy images or low-resolution images are in the same folder.

#### Step1

```python
python crop_single_file.py --path ./Test_images --task denoise 
```

#### Step 2

```python
python run.py -p test -c config/EMDiffuse-n.json -g 0 -b 60 --path ./Test_images/denoise_test_crop_patches
```

## vEMDiffuse for vEM isotropic reconstruction

### Training

Firstly, you should download volume data from Openorganelle or use your own vEM data. The data structure should be: 

```
Vem_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The nth layer
```

#### Training vEMDiffuse-i

```
python run.py -c config/vEMDiffuse-i.json -b 16 -gpu 0,1,2,3 --port 20022 --path ./Vem_data -z 5
```

Where parameter ``-z`` means the number of layers you want the model to learn to generate. For 8 nm * 8 nm * 48 nm, you should generate 5 layers to achieve 8 nm * 8 nm * 8nm voxel size.

####  Training vEMDiffuse-a

 ```
 python vEMa_pre.py --path ./Vem_data
 python run.py -c config/vEMDiffuse-a.json -b 16 -gpu 0,1,2,3 --port 20022 --path ./Vem_data/transposed -z 5
 ```

### Testing 

Firstly, you need to have an anisotropic volume with a data structure: 

```
Vem_test_data
	0.tif // The first layer
	1.tif // The second layer
	...
	n.tif // The nth layer
	
```

```
python run.py -p test -c config/vEMDiffuse-a.json -g 0 -b 16 --path ./Vem_test_data/ -z 5
```

The code will automatically help to crop the patches to fit the GPU memory. You need to stitch the patches into the whole volume. 
