import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import numpy as np

import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class CommentDataset(Dataset):
    def __init__(self, folder_path, train=False, test=False, valid=False):
        
        if (train==False and test==False and valid==False):
            raise Exception('One of the `train`, `test` or `valid` needs to be True, got `train = {}` `test = {}` and `valid = {}`'.format(train, test, valid))
        if (train==True and test==True and valid == True):
            raise Exception('Only one of the `train`, `test` or `valid` needs to be True, got `train = {}` `test = {}` and `valid = {}`'.format(train, test, valid))
        if (train==True and test==True):
            raise Exception('Only one of the `train` or `test` needs to be True, got `train = {}`, and `test = {}`'.format(train, test))
        if (train==True and valid==True):
            raise Exception('Only one of the `train` or `valid` needs to be True, got `train = {}`, and `valid = {}`'.format(train, valid))
        if (test==True and valid==True):
            raise Exception('Only one of the `test` or `valid` needs to be True, got `test = {}`, and `valid = {}`'.format(test, valid))

        self.train_df = None
        self.test_df = None
        self.valid_df = None

        # boolean values
        self.train = train
        self.test = test
        self.valid = valid

        self.data_selected = None
        self.comment_selected = None
    
        self.train_comment = []
        self.test_comment = []
        self.val_comment = []
    
        # Read the dataset
        self.data = pd.read_csv(folder_path, sep = ",")#.head(20)
        
        self.data = shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)
        
        self.data['Authors Biasness'].replace('LEFT', 0, inplace=True)
        self.data['Authors Biasness'].replace('RIGHT', 1, inplace=True)
        
        # split the dataset into train, test, and valid.
        self.train_df, test_df = train_test_split(self.data, test_size=0.2,  random_state=11)
        self.test_df, self.valid_df = train_test_split(test_df, test_size=0.5,  random_state=96)
        
        # a basic preprocessor, beeds to be done  outside the dataset function.
        #idx = [4, 7, 12, 19]
        #list_d = []
        #for i in range(len(self.data["comment"])):
        #    if i not in idx:
        #        d = self.data["comment"][i].split("', ")
        #        list_d.append(d)
        #    else:
        #        d = self.data["comment"][i].split("\", ")
        #        list_d.append(d)
        
        if self.train == True:
            # do the sorting
            # Sort the dataframe according to the number of comments on documents.
            self.train_df.sort_values(by=['Num of Comments'], ascending=False, inplace=True)       
            comments = []
            for com in self.train_df["Authors Comment"]:
                comments.append(com.split("-|-")[:-1])
            self.train_comment = comments
        elif self.test == True:
            # no need to sort
            comments = []
            for com in self.test_df["Authors Comment"]:
                comments.append(com.split("-|-")[:-1])
            self.test_comment = comments
        elif self.valid == True:
            # no need to sort
            comments = []
            for com in self.valid_df["Authors Comment"]:
                comments.append(com.split("-|-")[:-1])
            self.val_comment = comments
        
        
        # split the dataset into train, test, and valid.
        #self.train_df, test_df = train_test_split(self.data, test_size=0.2,  random_state=11)
        #self.test_df, self.valid_df = train_test_split(test_df, test_size=0.5,  random_state=96)

        #if self.train == True:
        # do the sorting
        #self.train_df["combined_text_len"] = 0
        #for i, row in self.train_df.iterrows():
        #    self.train_df.at[i, "combined_text_len"] = len(tokenizer(row["combined_text"]))
        # Sort the dataframe according to the length of the sentence
        #self.train_df.sort_values(by=['combined_text_len'], ascending=False, inplace=True)            
        #elif self.test == True:
            # no need ot sort
        #    self.test_df = self.test_df
        #elif self.valid == True:
            # no need ot sort
        #    self.valid_df = self.valid_df

    def __getitem__(self, idx):
        if self.train == True:
            self.data_selected = self.train_df
            self.comment_selected = self.train_comment

        elif self.test == True:
            self.data_selected = self.test_df
            self.comment_selected = self.test_comment

        elif self.valid == True:
            self.data_selected = self.valid_df
            self.comment_selected = self.val_comment

        label = self.data_selected.iloc[idx]["Authors Biasness"]
        sentence = self.comment_selected[idx]
        return sentence, label

    def __len__(self):
        
        
        if self.train == True:
            len_ = len(self.train_comment)

        elif self.test == True:
            len_ = len(self.test_comment)

        elif self.valid == True:
            len_ = len(self.val_comment)
        
        
        #if self.train == True:
        #len_ = len(self.comment)

        #elif self.test == True:
        #    len_ = len(self.test_df)

        #elif self.valid == True:
        #    len_ = len(self.valid_df)
        return len_

class InferenceCommentDataset(Dataset):
    def __init__(self, folder_path):
        
        self.train_df = None

        self.data_selected = None
        self.comment_selected = None
    
        self.train_comment = []
    
        # Read the dataset
        self.train_df = pd.read_csv(folder_path, sep = ",")#.head(20)
        #self.train_df = pd.read_csv(folder_path, lineterminator="\n")
        
        self.train_df = self.train_df.drop_duplicates(subset='Author Id')

        # do the sorting
        # Sort the dataframe according to the number of comments on documents.
        self.train_df.sort_values(by=['Num of Comments'], ascending=False, inplace=True)       
        comments = []
        for com in self.train_df["Authors Comment"]:
            comments.append(com.split("-|-")[:-1])
        self.train_comment = comments

    def __getitem__(self, idx):
    
        self.data_selected = self.train_df
        self.comment_selected = self.train_comment

        #label = self.data_selected.iloc[idx]["Annot"]
        sentence = self.comment_selected[idx]
        return sentence

    def __len__(self):  
        len_ = len(self.train_comment)

        return len_

class YoutubeBatchSampler(BatchSampler):
    def __init__(self, dataset, num_of_liberals, num_of_conservatives):
        
        self.label_list = []
        for _, label in dataset:
            self.label_list.append(label)

        self.label_list = torch.LongTensor(self.label_list) # list of all the labels in the dataset
        
        self.label_set = list(set(self.label_list.numpy())) # unique labels from the dataset

        self.label_to_indices = {label: np.where(self.label_list.numpy() == label)[0]
                                 for label in self.label_set}

        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0
        self.dataset = dataset
        self.num_of_liberals = num_of_liberals
        self.num_of_conservatives = num_of_conservatives
        self.batch_size = self.num_of_liberals + self.num_of_conservatives

    def __iter__(self):
        self.count = 0
        
        # maybe add <= 
        while self.count + self.batch_size < len(self.dataset):
            classes = np.array([1, 0])
            indices = []
            for class_ in classes:
                if(class_ == 0) :
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_] : self.used_label_indices_count[class_] + self.num_of_liberals])
                    self.used_label_indices_count[class_] += self.num_of_liberals

                    if self.used_label_indices_count[class_] + self.num_of_liberals > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
              
                elif(class_ == 1):
                    indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_] : self.used_label_indices_count[class_] + self.num_of_conservatives])
                    self.used_label_indices_count[class_] += self.num_of_conservatives

                    if self.used_label_indices_count[class_] + self.num_of_conservatives > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
              
              
            yield indices
            self.count = self.count + self.num_of_conservatives + self.num_of_liberals
            
    def __len__(self):
        return len(self.dataset) // self.batch_size


class CommentDataset_LSTM(Dataset):
    def __init__(self, folder_path, train=False, test=False, valid=False):
        
        if (train==False and test==False and valid==False):
            raise Exception('One of the `train`, `test` or `valid` needs to be True, got `train = {}` `test = {}` and `valid = {}`'.format(train, test, valid))
        if (train==True and test==True and valid == True):
            raise Exception('Only one of the `train`, `test` or `valid` needs to be True, got `train = {}` `test = {}` and `valid = {}`'.format(train, test, valid))
        if (train==True and test==True):
            raise Exception('Only one of the `train` or `test` needs to be True, got `train = {}`, and `test = {}`'.format(train, test))
        if (train==True and valid==True):
            raise Exception('Only one of the `train` or `valid` needs to be True, got `train = {}`, and `valid = {}`'.format(train, valid))
        if (test==True and valid==True):
            raise Exception('Only one of the `test` or `valid` needs to be True, got `test = {}`, and `valid = {}`'.format(test, valid))

        self.train_df = None
        self.test_df = None
        self.valid_df = None

        # boolean values
        self.train = train
        self.test = test
        self.valid = valid

        self.data_selected = None
        self.comment_selected = None
    
        self.train_comment = []
        self.test_comment = []
        self.val_comment = []
    
        # Read the dataset
        self.data = pd.read_csv(folder_path, sep = ",")#.head(20)
        
        self.data = shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)
        
        self.data['Authors Biasness'].replace('LEFT', 0, inplace=True)
        self.data['Authors Biasness'].replace('RIGHT', 1, inplace=True)
        
        # split the dataset into train, test, and valid.
        self.train_df, test_df = train_test_split(self.data, test_size=0.2,  random_state=11)
        self.test_df, self.valid_df = train_test_split(test_df, test_size=0.5,  random_state=96)
        
        if self.train == True:
            # do the sorting
            # Sort the dataframe according to the number of comments on documents.
            self.train_df.sort_values(by=['Num of Comments'], ascending=False, inplace=True)       
            comments = []
            for com in self.train_df["Authors Comment"]:
                comments.append(com.replace(" -|- ", "."))
            self.train_comment = comments
            
        elif self.test == True:
            # no need to sort
            comments = []
            for com in self.test_df["Authors Comment"]:
                comments.append(com.replace(" -|- ", "."))
            self.test_comment = comments
        elif self.valid == True:
            # no need to sort
            comments = []
            for com in self.valid_df["Authors Comment"]:
                comments.append(com.replace(" -|- ", "."))
            self.val_comment = comments

    def __getitem__(self, idx):
        if self.train == True:
            self.data_selected = self.train_df
            self.comment_selected = self.train_comment

        elif self.test == True:
            self.data_selected = self.test_df
            self.comment_selected = self.test_comment

        elif self.valid == True:
            self.data_selected = self.valid_df
            self.comment_selected = self.val_comment

        label = self.data_selected.iloc[idx]["Authors Biasness"]
        sentence = self.comment_selected[idx]
        return sentence, label

    def __len__(self):
        
        
        if self.train == True:
            len_ = len(self.train_comment)

        elif self.test == True:
            len_ = len(self.test_comment)

        elif self.valid == True:
            len_ = len(self.val_comment)

        return len_