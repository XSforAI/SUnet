# SUnet: A multi-organ segmentation network based on multiple attention

The official implementation of "SUnet: A multi-organ segmentation network based on multiple attention"

# install:

Recommended environment:

```python
    Python 3.8 + Pytorch 1.11.0 + torchvision 0.12.0
```


666666
Please use pip install -r requirements.txt to install the dependencies.

# Data preparation:
Synapse Multi-organ dataset: Sign up in the official Synapse website (https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the TransUNet's lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using python ./utils/preprocess_synapse_data.py or download the preprocessed data and save in the './data/synapse/' folder. Note: If you use the preprocessed data from TransUNet(https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py.

ACDC dataset: Download the preprocessed ACDC dataset from Google Drive of MT-UNet(https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and move into './data/ACDC/' folder.

Training:
......
