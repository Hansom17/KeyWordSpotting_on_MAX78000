import pandas

import numpy as np

import librosa

import torch

from torchvision import transforms

from torch.utils.model_zoo import tqdm

import ai8x

from PIL import Image

import os


usedDataset = 1



class MLCommonsTimeDomain(torch.utils.data.Dataset):

 

    def __init__(self,DataSetPath,test=False,transform=None,output2d=False) -> None:

        self.output2d = output2d
       
        self.transform = transform
        # DataSetPath = "mswc_microset/en/"

        ClipsPath = DataSetPath + "clips/"

        self.processedPath = DataSetPath + "processed/timeDomain"
 

        if test:  

            with open(DataSetPath + "en_test.csv") as file:

                csvData = pandas.read_csv(file, sep=',').sample(frac=1).reset_index(drop=True)

 

        else:

 

            with open(DataSetPath + "en_train.csv") as file:

                csvData = pandas.read_csv(file, sep=',').sample(frac=1).reset_index(drop=True)

 

        KeyWords = list(set(csvData['WORD']))
        
        
        if test:
            print("Loading TestSet")
            if os.path.exists(self.processedPath+"testData.pt"):
                self.data, self.labels = torch.load(self.processedPath+"testData.pt")
            else:
                self.data, self.labels = load_data(csvData,KeyWords,ClipsPath)
                try:
                    os.makedirs(self.processedPath)
                except:
                    print("processed exists")    
                self.saveData = (self.data,self.labels)
                torch.save(self.saveData,self.processedPath+"testData.pt")
        else:
            print("Loading TrainSet")
            if os.path.exists(self.processedPath+"trainData.pt"):
                self.data, self.labels = torch.load(self.processedPath+"trainData.pt")
            else:
                self.data, self.labels = load_data(csvData,KeyWords,ClipsPath)
                try:
                    os.makedirs(self.processedPath)
                except:
                    print("processed exists")    
                self.saveData = (self.data,self.labels)
                torch.save(self.saveData,self.processedPath+"trainData.pt")

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
 

    def __len__(self):

        return len(self.data)

 

    def __getitem__(self, index):

 

        img, target = self.data[index], self.labels[index]

        # if(self.output2d == True):
        #     img =  np.squeeze(img)
        
        # doing this so that it is consistent with all other datasets

        # to return a PIL Image

        #img = Image.fromarray(img, mode='L')

        
 

        if self.transform is not None:

            img = self.transform(img)
        
        #img = np.array(img)
        target = int(target)

        # print(type(img))
        # #print(f'MFCC index {index} has shape {img.shape}')
       
        # print(img.shape)
        return img, target

 

def load_data(traindata,keyWords,ClipsPath,sliced=False):

    

    x_list = []

    y_list = []

 

    totalSliceLength = 1 # Length to stuff the signals to, given in seconds

    # trainsize = len(traindata) # Number of loaded training samples

    # testsize = len(testdata) # Number of loaded testing samples

    dataSetSize = int(len(traindata) * usedDataset ) # Number of loaded training samples

    overlap = 128

 

    fs = 16000 # Sampling rate of the samples

    segmentLength = 1024 # Number of samples to use per segment

 

    sliceLength = (totalSliceLength * fs)

 

    for i in tqdm(range(dataSetSize)):

 

        sound_data, _ = librosa.load(ClipsPath+traindata['LINK'][i],res_type='kaiser_fast',sr=fs) # Read wavfile to extract amplitudes

 

        # plt.plot(train_sound_data)

 

        _x = sound_data.copy() # Get a mutable copy of the wavfile

 

        _x.resize(sliceLength) # Zero stuff the single to a length of sliceLength

 

        # _x_train = np.array([_x_train[i : i + segmentLength] for i in range(0, len(_x_train)-overlap, segmentLength-overlap)])

 

        x_list.append(_x.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain

 

        _y = keyWords.index(traindata['WORD'][i])

 

        y_list.append(_y)

 

    x_data = torch.from_numpy(np.asarray(x_list))

    y_data = torch.from_numpy(np.asarray(y_list))

 

    return x_data, y_data

 

# functions to convert audio data to image by mel spectrogram technique and augment data.

 


def mlcommosTD_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            transforms.ToTensor(),
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

def mlcommos_get_datasets2D(data, load_train=True, load_test=True):
    (data_dir, args) = data
    data_dir = "../mswc_microset/en/"
    transform = transforms.Compose([
            transforms.ToTensor(),
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
        'name': 'mlcommosTD',
        'input': (16000,1),
        'output': list(map(str, range(30))),
        'loader': mlcommosTD_get_datasets,
    }
]
