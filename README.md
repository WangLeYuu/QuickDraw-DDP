# QuickDraw-DDP
## Features

> **Transfer Learning:** Train a MobileNetv3 model using **340** classes from the QuickDraw dataset
>
> **Distributed Training:** Provide multi card training code


## Requirements

Required:

> matplotlib==3.9.2
> numpy==2.1.1
> onnx==1.16.1
> onnx_simplifier==0.4.36
> onnxruntime==1.18.1
> onnxsim==0.4.36
> Pillow==10.4.0
> scikit_learn==1.5.2
> seaborn==0.13.2
> torch==2.3.0
> torchsummary==1.5.1
> torchvision==0.18.0
> tqdm==4.66.4


You can install these dependencies via pip:

```python
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Download the dataset from here: [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/competitions/quickdraw-doodle-recognition)

This repository will use the **train_simplified** dataset provided by **Kaggle**

 1. Save all class **CSV** format files as **PNG** image format;
 2. Extract **10000** PNG format images from 340 categories for subsequent practice;
 3. Divide the training set, validation set, and testing set of 10000 pieces of data for each category into 8:1:1 ratios;

### 2. Get PNG format images
The following script can be used to convert CSV data into PNG image format for saving.

```python
python csv2png.py
```

> It should be noted that the drawing data is around 50 million, and the processing time is very long. It is recommended to run more scripts (PS: The code has added a statement to check whether the folder exists, so there is no need to worry about duplicate writing). You can also use the ```joblib``` library for multi-threaded acceleration (not recommended as it can easily crash if not played well).

> The image converted from train_Simplified to 256 * 256 size has **470G**; If there is not enough disk space, you can choose between 128 * 128 size or 64 * 64 size when converting PNG files. Single channel images can also be saved.


### 3. Check PNG format images

After processing, it is recommended to use the following script to check for any unprocessed categories.

```python
python check_class_num.py
```


### 4. Split Your Dataset

Run the following script to obtain the partitioned dataset, but you need to pay attention to modifying some paths.

```python
python split_datasets.py
```

### 5. Set Parameters
Define some parameters required for the future in ```option.py``` according to your conditions and requirements.


### 6. Define Data Generator
The Data Generator can be found in ```getdata.py```


### 7. Modify Network 

Define the model, this repository uses the small version of MobileNetv3. We need to change the output of the classifier layer of the model to the number of categories in ```model.py```.


### 8. Train Model

The following command can be used on the terminal to start multicard distributed training.

```python
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.8.89" --master_port=12345 train-DDP.py
```

The meanings of the relevant parameters are as follows:

- nproc_per_node: Number of graphics cards
- nnodes: number of machines
- node_rank: Machine Number
- masteraddr: Machine IP address
- master_port: Machine Port

By using the ```tensorboard --logdir=tensorboard_dir``` command on the terminal, you can view the training logs.


### 9. Model Transformation

The following script can be used to convert the PTH model to PTL format and ONNX format for mobile devices, making it easier to deploy the model on the client-side.

```python
python model_transfer.py
```


### 10. Model Evaluate

```evaluate.py``` provides two functions for predicting individual images and predicting folder images.

```python
python evaluate.py
```


## License

This project is licensed under the Apache 2.0 license. For detailed information, please refer to the LICENSE file.
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgement

> Kaggle Dataset: [Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/competitions/quickdraw-doodle-recognition)

> GitHub Dataset: [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

> CSDN: [王乐予-CSDN博客](https://blog.csdn.net/qq_42856191?type=blog)

