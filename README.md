# Image Descriptor with Visual Attention Mechanism Using Long Short-term Memory

This project is regarding image captioning. We follow most state-of-the-art networks based on Convolutional neural networks and Recurrent neural networks (CNN-RNN). We utilize CNN to extract features over the image, and then adopt RNN to generate captions from these features. To address the problem of object missing in the predicted text, we append a attention network to force the visual features to be considered at each time step. We present various configurations of CNN models and using merely LSTM for our RNN model. Throughout this project, we evaluate our networks on MS COCO dataset.

* Group Members: [Lin-Ying Cheng](https://github.com/lychengr3x), [Che-Ming Chia](https://github.com/shoachia), [Shang-Wei Hung](https://github.com/shangweihung), [Tsun-Hsu Lee](https://github.com/thlee-0810)

* The original code for image captioning is from: [pytorch-tutorial/image-captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning). We tweak it and add extra functions.

## Dataset

* [COCO](http://cocodataset.org/): COCO is a large-scale object detection, segmentation, and captioning dataset.  

## Testing Environment  

* Pytorch version: `1.1.0`
* CUDA version: `9.0.176`
* Python version: `3.6.8`
* CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
* GPU: GeForce GTX 1080 Ti (11172MB GRAM)
* RAM: 32GB

## Code Organization

```
download_dataset.sh      --  Download COCO ataset, including images and captions
src/build_vocab.py       --  Build a vocabulary wrapper of COCO dataset captions
src/data.py              --  Module for preprocessing the images
src/demo_training.ipynb  --  Run a demo of training a model
src/demo_testing.ipynb   --  Run a demo of testing a model
src/main.py              --  Main file that you could run it in terminal
src/model.py             --  Models of CNN and RNN
src/resize.py            --  Module for resizing the images
src/utils.py             --  Useful functions and our ImageDescriptor model
```

## Usage

### 1. Install required packages

```bash
pip install -r requirements.txt --user  
```

### 2. Install COCO API  

```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install --user
cd ../../
git clone https://github.com/lychengr3x/Image-Descriptor.git
cd Image-Descriptor
```

### 3. Download Dataset

If you want to use preprocessed dataset, you can skip this step.

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

### 4. Preprocess the data

You can do it **from the scratch**

```bash
cd src
# training set
python build_vocab.py  
python resize.py

# validation set
python build_vocab.py --caption_path='../data/annotations/captions_val2014.json' --vocab_path='../data/vocab_val.pkl'
python resize.py --image_dir='../data/val2014/'
```

, or simply download **preprocessed dataset and trained model**.

* `annotations`: This directory includes two files, `captions_train2014.json` and `captions_val2014.json`. [(link)](https://drive.google.com/file/d/1KrNtlg5-Z11abTR50iBuIxpPYcS1EjJf/view?usp=sharing)
  
* `vocab`: This includes vocabulary of training set and validation set, `vocab.pkl` and `vocab_val.pkl`. [(link)](https://drive.google.com/file/d/1D4ZeIju-Min-S9BqAh2Odr39MCSsZGty/view?usp=sharing)

* `resized2014`: This directory includes all resized images (`256x256`) of training set and validation set. [(link)](https://drive.google.com/file/d/1B-q-ZInOvUFntRPq30CXee89o1tB9WPA/view?usp=sharing)

### 5. Train the model in the background save log file  

It takes around 30 minutes.

```bash  
# no attention layer
nohup python main.py --mode='train' > log.txt &  

# with attention layer
nohup python main.py --mode='train' --attention=True > log.txt &  
```

**How to specify a model**:

Take `resnet152` for example. Assign `--encoder=resnet` and `--encoder_ver=152`.

* Here are 2 of the pretrained model `resnet101`: `resnet101-epoch-7.ckpt` [(link)](https://drive.google.com/file/d/1WTss11jFJdoZ6XUxNTW8aL-zYlJsi1X1/view?usp=sharing), `resnet101-epoch-15.ckpt` [(link)](https://drive.google.com/file/d/1pSCmjnc_5PDJwD4aAiuAqQ0KA5Lv8DjW/view?usp=sharing)

* Here is a demo that shows how to train in the jupyter notebook: `demo_training.ipynb` [(link)](src/demo_training.ipynb)

### 6. Evaluate the model

To get a caption for a specific image.

```bash
# no attention layer
python main.py --mode=test --encoder=resnet --encoder_ver=101 --image_path=../png/example.png --model_dir=../models --checkpoint=resnet101-epoch-7.ckpt

# with attention layer
python main.py --mode=test --encoder=resnet --encoder_ver=101 --attention=True --image_path=../png/example.png --model_dir=../models --checkpoint=resnet101-epoch-7.ckpt
```

To get the loss of validation set at specific epoch. (run in the background).  
It takes around 20 minutes.

```bash
# no attention layer
nohup python main.py --mode=val --encoder=resnet --encoder_ver=101 --model_dir=../models --checkpoint=epoch-7.ckpt > val_loss.txt &

# with attention layer
nohup python main.py --mode=val --encoder=resnet --encoder_ver=101 --attention=True --model_dir=../models --checkpoint=epoch-7.ckpt > val_loss_att.txt &
```

* Here is a demo that shows how to test in the jupyter notebook: `demo_testing.ipynb` [(link)](src/demo_testing.ipynb)

## File arrangement

If you want to re-run the `demo_testing.ipynb` [(link)](src/demo_testing.ipynb) directly, make sure you download files from the above links and put them in the right place, shown as the following.
```
.
|--- png/
     |--- example.png
     |--- test_01_resize.jpg
     |--- test_02_resize.jpg
     |--- test_03_resize.jpg
     |--- test_04_resize.jpg
|--- src/
     |--- build_vocab.py
     |--- data.py
     |--- demo_training.ipynb
     |--- demo_testing.ipynb
     |--- main.py
     |--- model.py
     |--- resize.py
     |--- utils.py
|--- models/
     |--- config-resnet101.txt
     |--- resnet101-epoch-7.ckpt
     |--- resnet101-epoch-15.ckpt
|--- data/
     |--- resized2014/
     |--- annotations/
          |--- captions_train2014.json
          |--- captions_val2014.json
     |--- vocab.pkl
     |--- vocab_val.pkl
```
