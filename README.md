# Image-Descriptor  

Image captioning

* Group Members: [Che-Ming Chia](https://github.com/shoachia), [Shang-Wei Hung](https://github.com/shangweihung), [Tsun-Hsu Lee](https://github.com/thlee-0810)

The original code for image captioning is from: [pytorch-tutorial/image-captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning). We tweak it and add extra functions.

## Dataset

* [COCO](http://cocodataset.org/): COCO is a large-scale object detection, segmentation, and captioning dataset.  

## Testing Environment  

* Pytorch version: `1.0.0`
* CUDA version: `9.0.176`
* Python version: `3.6.8`
* CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz
* GPU: GeForce GTX 1080 Ti (11172MB GRAM)
* RAM: 32GB

## Usage

1. Install required packages

```bash
pip install -r requirements.txt --user  
```

2. Install COCO API  

```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install --user
```

3. Download Dataset

```bash
cd ../../
git clone https://github.com/lychengr3x/Image-Descriptor.git
cd Image-Descriptor
chmod +x download_dataset.sh
./download_dataset.sh
```

4. Preprocessing

```bash
cd src
# training set
python build_vocab.py  
python resize.py

# validation set
python build_vocab.py --caption_path='../data/annotations/captions_val2014.json' --vocab_path='../data/vocab_val.pkl'
python resize.py --image_dir='../data/val2014/'
```

5. Train the model in the background save log file  

```bash  
# no attention layer
nohup python run.py --mode='train' > log.txt &  

# with attention layer
nohup python run.py --mode='train' --attention=True > log.txt &  
```

6. Evaluate the model  

```bash
# no attention layer
python run.py --mode='eval' --image_path='../png/example.png'

# with attention layer
python run.py --mode='eval' --attention=True --image_path='../png/example.png'
```
