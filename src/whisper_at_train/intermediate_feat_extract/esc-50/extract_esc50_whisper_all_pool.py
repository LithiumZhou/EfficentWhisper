# -*- coding: utf-8 -*-
# @Time    : 1/19/23 11:35 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : extract_esc50.py

# extract representation for all layers for whisper model, pool by 10, not include the input mel.
# save as npz to save space

import json
import torch
import os
os.environ["XDG_CACHE_HOME"] = './'
import whisper_at
import numpy as np
from whisper_at.model import Whisper, ModelDimensions
import skimage.measure
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--split", type=int, default=0, help="which split")
args = parser.parse_args()

def extract_audio(dataset_json_file, mdl, tar_path):
    if os.path.exists(tar_path) == False:
        os.mkdir((tar_path))
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        for idx, entry in enumerate(data):
            wav = entry["wav"]

            if os.path.exists(tar_path + '/' + wav.split('/')[-1][:-4] + 'npz') == False:
                # NOTE: this use a customized whisper model for feature extraction, original whisper model does not have transcribe_audio function
                _, audio_rep = mdl.transcribe_audio(wav)
                audio_rep = audio_rep[0]
                audio_rep = torch.permute(audio_rep, (2, 0, 1)).detach().cpu().numpy()
                # 下采样十倍 ，原本是 500，1280这样 ，(1,10,1)代表在第二个维度上每十个时间帧平均为一个
                audio_rep = skimage.measure.block_reduce(audio_rep, (1, 5, 1), np.mean) # downsample x10 for esc, 20 for audioset
                audio_rep = audio_rep[1:]
                np.savez_compressed(tar_path + '/' + wav.split('/')[-1][:-3] + 'npz', audio_rep)

mdl_size_list = ['large-v1']
# mdl_size_list = ['large-v2', 'large-v1', 'medium.en', 'medium', 'small.en', 'small', 'base.en', 'base', 'tiny.en', 'tiny'] 
for mdl_size in mdl_size_list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path = '/root/.cache/whisper/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims) # NOTE: this use a customized whisper model for feature extraction, original whisper model does not have transcribe_audio function
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    # tar_path即存在哪
    tar_path = '/g813_u1/zhoujia/whisper-at/data/' + 'whisper_' + mdl_size + '_tdim=50/'
    esc_train1 = '/g813_u1/zhoujia/whisper-at/data/datafiles/esc_train_data_1.json'
    esc_eval1 = '/g813_u1/zhoujia/whisper-at/data/datafiles/esc_eval_data_1.json' # esc-50 is 5-fold cross-validation, so 1st train and eval split covers all datas
    extract_audio(esc_train1, model, tar_path)
    extract_audio(esc_eval1, model, tar_path)
    del model