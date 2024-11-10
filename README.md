# Introduce

This repository provides methods for extracting Whisper features and examples for training the ESC-50 dataset based on Whisper-AT. This means you can use Whisper to extract features from any dataset and train it.

## Qucikstart

1. Prepare your environment

```java
conda create --name whisper python=3.9
pip install -r requirements.txt 
```

2. Then download the characteristics of esc-50 [here](https://www.dropbox.com/s/hmmdopfjlq3o3vs/esc_feat.zip?dl=1). And you need to replace lines 84-89 in [run_esc.sh ](https://github.com/LithiumZhou/EfficentWhisper/blob/main/src/whisper_at_train/esc-50/run_esc.sh)with your dataset json path,label.csv and feature path. datapath json and label.csv are ready for you.

3. Get the pre-trained model,The pre-trained models that can be used in this step are available [here](https://github.com/LithiumZhou/EfficentWhisper/tree/main/pretrained_models)

```java
python get_model.py
```

4. start train 

```java
cd src/whisper_at_train/esc/
chmod +x run_esc.sh
./run_esc.sh
```

## Extracting features using Whisper

The feature extraction code can be run directly. You just need to prepare your dataset and modify lines 27, 52-53 [here](https://github.com/LithiumZhou/EfficentWhisper/blob/main/src/whisper_at_train/intermediate_feat_extract/extract_esc_whisper.py).The specific meaning of the code is to prepare a json file with the storage path of all your data sets.

