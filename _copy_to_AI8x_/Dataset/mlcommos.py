import math
import pandas

import numpy as np

import librosa

import torch

from torchvision import transforms

from torch.utils.model_zoo import tqdm

import ai8x

from PIL import Image

import os

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


usedDataset = 1

numberOfWords = 2

words = ['left','right']

pretrain = True

class MLCommonsTimeDomainToFreqDomain(torch.utils.data.Dataset):

 

    def __init__(self,DataSetPath,test=False,transform=None,output2d=False) -> None:

        self.output2d = output2d
       
        self.transform = transform
        # DataSetPath = "mswc_microset/en/"

        ClipsPath = DataSetPath + "clips/"

        self.processedPath = DataSetPath + "processed/time2four/"
 

        if test:  

            with open(DataSetPath + "en_test.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        else:

 

            with open(DataSetPath + "en_train.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        KeyWords = list(set(csvData['WORD']))
        
        
        if test:
            print("Loading TestSet")
            if not os.path.exists(self.processedPath+"test/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"test/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"test",ClipsPath)
            self.data, self.mfccs = self.load_data(csvData,"test")
                   
        else:
            print("Loading TrainSet")
            if not os.path.exists(self.processedPath+"train/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"train/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"train",ClipsPath)
            self.data, self.mfccs = self.load_data(csvData,"train")

        print(self.data[0].shape)
        print(self.mfccs[0].shape)
        

    def __len__(self):

        return len(self.data)

 

    def __getitem__(self, index):

        img, target = self.data[index], self.mfccs[index].swapaxes(0,1)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        
        target = torch.from_numpy(target).type(torch.FloatTensor)
 
        if self.transform is not None:

            img = self.transform(img)

        
        return img, target

    def load_data(self,traindata,subset):
        x_list = []

        y_list = []
        
        dataSetSize = int(len(traindata) * usedDataset )
        
        for i in tqdm(range(dataSetSize)):
            data = np.load(self.processedPath+subset+f"/data{i}.npz")
            signal = data["x_data"]
            slice_length = 256 # in seconds
            overlap = 0 # in seconds
            slices = np.arange(0, len(signal), slice_length-overlap, dtype=np.int)

            x_data = []
            for start, end in zip(slices[:-1], slices[1:]):
                start_audio = start
                end_audio = (end + overlap)
                x_data.append(signal[int(start_audio): int(end_audio)])
            x_data = np.asarray(x_data)
            x_list.append(np.resize(x_data,(64,256)))
            y_list.append(np.resize(data["y_data"],(128,64)))
            
        return x_list, y_list
        

    def get_data(self,traindata,subset,ClipsPath,sliced=True):
        

        totalSliceLength = 1 # Length to stuff the signals to, given in seconds

        dataSetSize = int(len(traindata) * usedDataset ) # Number of loaded training samples

        overlap = 0

        fs = 16000 # Sampling rate of the samples

        segmentLength = 256 # Number of samples to use per segment

        numOfSegments = math.ceil((totalSliceLength*fs-overlap)/(segmentLength-overlap))

        sliceLength = segmentLength * numOfSegments

        n_mels=128
        
        f_max=8000
        
        hop_length=256
        
        n_fft=256
        
        
        for i in tqdm(range(dataSetSize)):

            sound_data, _ = librosa.load(ClipsPath+traindata['LINK'][i],res_type='kaiser_fast',sr=fs) # Read wavfile to extract amplitudes

            _x = sound_data.copy() # Get a mutable copy of the wavfile
            
            _x.resize(sliceLength) # Zero stuff the single to a length of sliceLength
            
            _y = librosa.feature.melspectrogram(y=_x, sr=fs, n_mels=n_mels, fmax=f_max,hop_length=hop_length, n_fft=n_fft)

            _x = np.asarray(_x,dtype=np.float16)

            
            
            np.savez_compressed(self.processedPath+subset+f"/data{i}.npz",x_data=np.asarray(_x),y_data=np.asarray(_y))

    

        return 


class MLCommons(torch.utils.data.Dataset):

 

    def __init__(self,DataSetPath,test=False,transform=None,output2d=False) -> None:

        self.output2d = output2d
       
        self.transform = transform
        # DataSetPath = "mswc_microset/en/"

        ClipsPath = DataSetPath + "clips/"

        self.processedPath = DataSetPath + "processed/mfcc/"
        
        if pretrain:
            self.processedPath = DataSetPath + "processed/pretrain/mfcc/"
 
 

        if test:  

            with open(DataSetPath + "en_test.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        else:

 

            with open(DataSetPath + "en_train.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        self.keyWords = ['right', 'forward', 'dog', 'bed', 'five', 'stop', 'off', 'down', 'house', 'zero', 'marvin', 'bird', 'wow', 'left', 'eight', 'visual', 'seven', 'yes', 'two', 'six', 'three', 'backward', 'tree', 'follow', 'four', 'happy', 'learn', 'one', 'nine', 'cat', 'sheila']

        
        if pretrain:
            csvData = csvData.loc[(csvData['WORD'] == 'left') | (csvData['WORD'] == 'right')].reset_index()
            print(csvData)
        
        if test:
            print("Loading TestSet")
            if not os.path.exists(self.processedPath+"test/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"test/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"test",ClipsPath)
            self.data, self.labels = self.load_data(csvData,"test")
                   
        else:
            print("Loading TrainSet")
            if not os.path.exists(self.processedPath+"train/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"train/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"train",ClipsPath)
            self.data, self.labels = self.load_data(csvData,"train")

        print(self.data[0].shape)
        # print(self.labels[0].shape)
        

    def __len__(self):

        return len(self.data)

 

    def __getitem__(self, index):

        img, target = self.data[index], self.labels[index]
        
        if not self.output2d:
            img = np.expand_dims(img,0)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        
        target = int(target)
 
        if self.transform is not None:

            img = self.transform(img)

        
        return img, target

    def load_data(self,traindata,subset):
        x_list = []

        y_list = []
        
        dataSetSize = int(len(traindata) * usedDataset )
        
        for i in tqdm(range(dataSetSize)):
            data = np.load(self.processedPath+subset+f"/data{i}.npz")
            x_list.append(np.resize(data["x_data"],(128,64)))
            index = int(data["y_data"])
            if pretrain:
                y = words.index(self.keyWords[index])
            else:
                if self.keyWords[index] in words:
                    y = words.index(self.keyWords[index])
                    
                else:
                    y = len(words)
            
            
            
            y_list.append(y)
            
        return x_list, y_list
        

    def get_data(self,traindata,subset,ClipsPath,sliced=True):
        

        totalSliceLength = 1 # Length to stuff the signals to, given in seconds

        dataSetSize = int(len(traindata) * usedDataset ) # Number of loaded training samples

        overlap = 0

        fs = 16000 # Sampling rate of the samples

        segmentLength = 256 # Number of samples to use per segment

        sliceLength = totalSliceLength * fs

        n_mels=128
        
        f_max=8000
        
        hop_length=256
        
        n_fft=256
        
        
        for i in tqdm(range(dataSetSize)):

            sound_data, _ = librosa.load(ClipsPath+traindata['LINK'][i],res_type='kaiser_fast',sr=fs) # Read wavfile to extract amplitudes

            _x = sound_data.copy() # Get a mutable copy of the wavfile
            
            _x.resize(sliceLength) # Zero stuff the single to a length of sliceLength
            
            # step = segmentLength - overlap
            # _x = [_x[j : j + segmentLength] for j in range(0, len(_x), step)]

            _x = librosa.feature.melspectrogram(y=_x, sr=fs, n_mels=n_mels, fmax=f_max,hop_length=hop_length, n_fft=n_fft)

            _y = self.keyWords.index(traindata['WORD'][i])

            
            np.savez_compressed(self.processedPath+subset+f"/data{i}.npz",x_data=np.asarray(_x),y_data=np.asarray(_y))

    

        return 


class MLCommonsTimeDomain(torch.utils.data.Dataset):

    

    def __init__(self,DataSetPath,test=False,transform=None,output2d=False) -> None:
        
        self.fs = 16000
        
        
        self.output2d = output2d
       
        self.transform = transform
        # DataSetPath = "mswc_microset/en/"

        ClipsPath = DataSetPath + "clips/"

        self.processedPath = DataSetPath + "processed/time/"
        
        if pretrain:
            self.processedPath = DataSetPath + "processed/pretrain/time/"
 

        if test:  

            with open(DataSetPath + "en_test.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        else:

 

            with open(DataSetPath + "en_train.csv") as file:

                csvData = pandas.read_csv(file, sep=',')

 

        self.keyWords = ['right', 'forward', 'dog', 'bed', 'five', 'stop', 'off', 'down', 'house', 'zero', 'marvin', 'bird', 'wow', 'left', 'eight', 'visual', 'seven', 'yes', 'two', 'six', 'three', 'backward', 'tree', 'follow', 'four', 'happy', 'learn', 'one', 'nine', 'cat', 'sheila']

        print(self.keyWords)
        
        if pretrain:
            csvData = csvData.loc[(csvData['WORD'] == 'left') | (csvData['WORD'] == 'right')].reset_index()
            print(csvData)
        
        
        if test:
            print("Loading TestSet")
            if not os.path.exists(self.processedPath+"test/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"test/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"test",ClipsPath)
            self.data, self.labels = self.load_data(csvData,"test")
                   
        else:
            print("Loading TrainSet")
            if not os.path.exists(self.processedPath+"train/data0.npz"):
                try:
                    os.makedirs(self.processedPath+"train/")
                except:
                    print("processed exists") 
                self.get_data(csvData,"train",ClipsPath)
            self.data, self.labels = self.load_data(csvData,"train")
        

    def __len__(self):

        return len(self.data)

 

    def __getitem__(self, index):

        img, target = self.data[index], int(self.labels[index])

        img = torch.from_numpy(img).type(torch.FloatTensor)
        
 
        if self.transform is not None:

            img = self.transform(img)

        
        return img, target

    def load_data(self,traindata,subset):
        x_list = []

        y_list = []
        
        dataSetSize = int(len(traindata) * usedDataset )
        
        for i in tqdm(range(dataSetSize)):
            data = np.load(self.processedPath+subset+f"/data{i}.npz")
            signal = data["x_data"]
            slice_length = 128 # in seconds
            overlap = 64 # in seconds
            slices = np.arange(0, len(signal), slice_length-overlap, dtype=np.int)

            x_data = []
            for start, end in zip(slices[:-1], slices[1:]):
                start_audio = start
                end_audio = (end + overlap)
                x_data.append(signal[int(start_audio): int(end_audio)])
            x_data = np.asarray(x_data)
            x_list.append(x_data)
            
            index = int(data["y_data"])
            if pretrain:
                y = words.index(self.keyWords[index])
            else:
                if self.keyWords[index] in words:
                    y = words.index(self.keyWords[index])
                    
                else:
                    y = len(words)
            
            
            
            y_list.append(y)
            
        return x_list, y_list
        

    def get_data(self,traindata,subset,ClipsPath,sliced=True):
        

        totalSliceLength = 1 # Length to stuff the signals to, given in seconds

        dataSetSize = int(len(traindata) * usedDataset ) # Number of loaded training samples


        sliceLength = totalSliceLength * self.fs
        
        
        for i in tqdm(range(dataSetSize)):

            sound_data, _ = librosa.load(ClipsPath+traindata['LINK'][i],res_type='kaiser_fast',sr=self.fs) # Read wavfile to extract amplitudes

            _x = sound_data.copy() # Get a mutable copy of the wavfile
            
            _x.resize(sliceLength) # Zero stuff the single to a length of sliceLength

            _x = np.asarray(_x,dtype=np.float16)

            _y = self.keyWords.index(traindata['WORD'][i])

            
            np.savez_compressed(self.processedPath+subset+f"/data{i}.npz",x_data=np.asarray(_x),y_data=np.asarray(_y))

    

        return 



# functions to convert audio data to image by mel spectrogram technique and augment data.

def mlTD_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

    if load_train:
        train_dataset = MLCommonsTimeDomain(data_dir, test=False ,transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MLCommonsTimeDomain(data_dir, test=True ,transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

 

    return train_dataset, test_dataset


def mlTDtoFD_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

    if load_train:
        train_dataset = MLCommonsTimeDomainToFreqDomain(data_dir, test=False ,transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MLCommonsTimeDomainToFreqDomain(data_dir, test=True ,transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

 

    return train_dataset, test_dataset

def mlcommos_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

    if load_train:
        train_dataset = MLCommons(data_dir, test=False ,transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MLCommons(data_dir, test=True ,transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

 

    return train_dataset, test_dataset


def mlcommos_get_datasets2d(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

    if load_train:
        train_dataset = MLCommons(data_dir, test=False ,transform=transform,output2d=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MLCommons(data_dir, test=True ,transform=transform,output2d=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

 

    return train_dataset, test_dataset

"""

Dataset description

"""

datasets = [
    {
        'name': 'mltdtofd',
        'input': (64,256),
        'output': (64,128),
        'loader': mlTDtoFD_get_datasets,
    },
    {
        'name': 'mltd',
        'input': (249,128),
        'output': list(map(str, range(len(words)+1))),
        'weight': [1.33,1,0.05],
        'loader': mlTD_get_datasets,
    },
    {
        'name': 'mlcommos',
        'input': (1,128,64),
        'output': list(map(str, range(len(words)+1))),
        'loader': mlcommos_get_datasets,
    },
    {
        'name': 'mlcommos2d',
        'input': (128, 64),
        'output': list(map(str, range(len(words)+1))),
        'loader': mlcommos_get_datasets2d,
    }
]
