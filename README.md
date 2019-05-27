# Image-Descriptor  

Image captioning

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

```
$ python -c "import nltk; nltk.download('punkt')"
$ python build_vocab.py   
$ python resize.py
```

5. Train the model in the background save log file  

```
$ nohup python train.py > log.txt &  
```

6. Test the model  

```
$ python sample.py --image='png/example.png' 
```
