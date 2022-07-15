import torch
from torch.utils.data import Dataset, DataLoader, random_split

import re
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
        
        self.data['Annot'].replace('LEFT', 0, inplace=True)
        self.data['Annot'].replace('RIGHT', 1, inplace=True)
        
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
            self.train_df.sort_values(by=['Number of Comment'], ascending=False, inplace=True)       
            comments = []
            for com in self.train_df["comment"]:
                comments.append(com.split("-|-")[:-1])
            self.train_comment = comments
        elif self.test == True:
            # no need to sort
            comments = []
            for com in self.test_df["comment"]:
                comments.append(com.split("-|-")[:-1])
            self.test_comment = comments
        elif self.valid == True:
            # no need to sort
            comments = []
            for com in self.valid_df["comment"]:
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

        label = self.data_selected.iloc[idx]["Annot"]
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