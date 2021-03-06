# MHPM
The code for the paper Multi-Human Parsing Machines

## Data
The dataset used in this repo is MHP-v1, which can be downloaded from our [LV-MHP](https://lv-mhp.github.io/dataset) website. 

Put the downloaded data under folder

```basj
./data
```
and unzip it to generate two subfolders: images and annotations.
Run the scripts 
```python
generate_global_seg.py
generate_global_tag.py
```
inside ./data to generate the global segmentation maps and global tags. 

## Models
Models for deployment are [here](https://drive.google.com/file/d/1x_aDKi-A0-C0cmmqIKH6ekhMITKwIFCr/view?usp=sharing)

Pre-trained models are [here](https://drive.google.com/file/d/12QdisT0SKSsP4qPb_tEBYLVFrZorfxF8/view?usp=sharing)


### Ackonwledgement:
Deeplab v2 in pytorch: https://github.com/speedinghzl/Pytorch-Deeplab

fastSceneUnderstanding: https://github.com/DavyNeven/fastSceneUnderstanding

Dense CRF: https://github.com/lucasb-eyer/pydensecrf

