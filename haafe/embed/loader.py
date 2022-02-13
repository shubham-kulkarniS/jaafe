import numpy as np
import pandas as pd
import scipy

import os
import json
from functools import partial
from tqdm import tqdm
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

COUGHVID = "/home/shubham/datasets/coughvid/public_dataset"

metadf = pd.read_csv(os.path.join(COUGHVID,"all_metadata.csv"))
# useful_df = metadf[~metadf["status"].isna()] # and metadf["cough_detected"]>0.65]
useful_df = pd.read_csv(os.path.join(COUGHVID,"use_metadata.csv"))


vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish.cuda().eval()


def vggish_reader(path):
    x = vggish(path).detach()
    if x.ndim > 1:
        x = x.mean(dim=0)
    return x


def make_dataset(metadata,audio_read = vggish_reader,label_read = {'positive':0,'negative':1}):
    flist = metadata["uuid"]
    labels = metadata["Labels"]
    indices = list(metadata.index)
    x,y = [],[]

    for i in tqdm(indices):
        f = os.path.join(COUGHVID,flist[i]+'.wav')
        x.append(audio_read(f).cpu().detach().numpy())
        y.append(label_read[labels[i]])
    return np.array(x),np.array(y)


# The PASE model use the raw (padded) signal as input 
def audio_reader(path, max_seconds):
    # y, sfr = wav_read(path)
    sr, y = scipy.io.wavfile.read(path)
    y = y/32768 # Normalize to -1..1
    y = y.astype(np.float32)
    # if len(y) > 16000*max_seconds:
    #    print(path, ':', len(y)/16000, 'seconds')
    y.resize(16000*max_seconds) # Ten seconds with zero padding
    return y

def audio_labeler(file_name,df):
    return 0



# PyTorch Dataset
class Loader(data.Dataset):

    def __init__(self, metadata, max_seconds=20):

        # classes, weight, class_to_id = get_classes()
        # self.classes = classes
        self.weight = None
        # self.class_to_id = {label: 2*i-1 for i, label in enumerate(classes)}
        self.df = metadata

        self.audio_read = partial(audio_reader,max_seconds=max_seconds)
        self.label_read = {'positive':1,'negative':-1}


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (key, params, target) where target is class_index of the target class.
        """
        info = self.df.iloc[index]
        key = info["uuid"]
        filename = info["uuid"] + '.wav'
        label = info["Labels"]
        path = os.path.join(COUGHVID , filename)

        audio = self.audio_read(path)
        target = self.label_read[label]
        return key, audio, target

    def __len__(self):
        return len(self.df)





def wav_read(path):
    sr, y = scipy.io.wavfile.read(path)
    y = y/32768 # Normalize to -1..1
    return y, sr


# Create a dataset with (key, wave_file, target_id) entries
## ?????????????
# def make_dataset(kaldi_path, class_to_id):
#     text_path = os.path.join(kaldi_path, 'text')     # labels
#     wav_path = os.path.join(kaldi_path, 'wav.scp')   # audio files

#     key_to_word = dict()
#     key_to_wav = dict()
    
#     with open(wav_path, 'rt') as wav_scp:
#         for line in wav_scp:
#             key, wav = line.strip().split(' ', 1)
#             key_to_wav[key] = wav
#             key_to_word[key] = None # default

#     if os.path.isfile(text_path):
#         with open(text_path, 'rt') as text:
#             for line in text:
#                 key, word = line.strip().split(' ', 1)
#                 key_to_word[key] = word

#     wavs = []
#     for key, wav_command in key_to_wav.items():
#         word = key_to_word[key]
#         word_id = class_to_id[word] if word is not None else -1 # default for test
#         wav_item = [key, wav_command, word_id]
#         wavs.append(wav_item)

#     return wavs


## prepare dataset :-|
def coughvid_dataset():
    flist = os.listdir(COUGHVID)
    json_list = [f for f in flist if f.endswith(".json")]
    wav_list = [f for f in flist if f.endswith(".wav")]
    metalist = []
    for j in json_list:
        js = os.path.join(COUGHVID,j)
        with open(js) as f:
            temp = json.load(f)
            temp['uuid'] = j[:-5]
            metalist.append(temp)
    dfo = pd.DataFrame(metalist)
    return dfo ## saved as all metadat csv flie


