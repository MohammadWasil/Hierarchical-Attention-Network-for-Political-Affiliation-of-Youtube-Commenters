import os
import argparse
import yaml

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import HierarchicalAttentionNetwork, UserClassificationModel
from datasets import CommentDataset, YoutubeBatchSampler, CommentDataset_LSTM
from data_utils import collate_batch, collate_batch_LSTM, vocabulary, save_vocabulary, load_vocabulary, get_vocabulary, get_embedding_matrix
from Evaluate import evaluate
from Train import train
from plot import plot
from model_utils import load_model

data_folder = "12. Subscription Training Data.csv"

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--MODEL', dest='MODEL',
        choices=['HAN', 'LSTM'],
        help='{HAN, LSTM}',
        type=str, required=True
    )

    parser.add_argument(
        '--RUN_MODE', dest='RUN_MODE',
        choices=['train', 'test'],
        help='{train, test}',
        type=str, required=True
    )

    parser.add_argument(
        '--CPU', dest='CPU',
        help='use CPU instead of GPU',
        action='store_true'
    )

    parser.add_argument(
        '--TRAINED_MODEL', dest='TRAINED_MODEL',
        help='upload trained model path',
        type=int
    )

    args = parser.parse_args()
    return args


class MainExec(object):
    def __init__(self, args, configs):
        self.args = args
        self.cfgs = configs

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # for failsafe
        #if self.args.RUN_MODE == 'test' or self.args.RUN_MODE =='bleu':
        #    if self.args.MODEL == None:
        #        raise Exception('Add a model number you need to evaluate, e.g Model_8.pickle, then pass 8 as an argument')

    def run(self, args):
        BATCH_SIZE = self.cfgs["BATCH_SIZE"]
        learning_rate = self.cfgs["lr"]

        vocab_size = 0
        
        # can change this accordingly. on CFG
        USE_PRETRAINED_EMBEDDING_MATRIX = self.cfgs["USE_PRETRAINED_EMBEDDING_MATRIX"]

        embedding_size = self.cfgs["EMBEDDING_SIZE"]
        num_class = self.cfgs["NUM_CLASS"]
        EPOCHS = self.cfgs["EPOCHS"]

        if self.args.RUN_MODE == "train":
            print("Training")

            if self.args.MODEL == "HAN":

                # create dataset loader for train and evaluation set.
                dataset_train = CommentDataset(data_folder, train=True, test=False, valid=False)
                dataset_valid = CommentDataset(data_folder, train=False, test=False, valid=True)

                if self.cfgs["BATCH_SAMPLER"] == False:
                    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                                shuffle=False, collate_fn=collate_batch)
                    val_dataloader = DataLoader(dataset_valid,
                                                shuffle=False, collate_fn=collate_batch)
                elif self.cfgs["BATCH_SAMPLER"] == True:
                    num_of_liberals = int(BATCH_SIZE / 2)
                    num_of_conservatives = int(BATCH_SIZE - num_of_liberals)

                    batch_sampler_train = YoutubeBatchSampler(dataset_train, num_of_liberals, num_of_conservatives)

                    train_dataloader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                                collate_fn=collate_batch)
                    val_dataloader = DataLoader(dataset_valid,
                                                shuffle=False, collate_fn=collate_batch)

                # create/load vocabulary
                PATH_TO_VOCABULARY = self.cfgs["PATH_TO_SAVE_VOCAB_HAN"]
                vocab, vocab_size = get_vocabulary(PATH_TO_VOCABULARY, "HAN", dataset_train)
                
                embedding_matrix = get_embedding_matrix(USE_PRETRAINED_EMBEDDING_MATRIX, vocab)

                ENCODER_HIDDEN_DIMENSION = self.cfgs["ENCODER_HIDDEN_DIM_HAN"]
                DECODER_HIDDEN_DIMENSION = self.cfgs["DECODER_HIDDEN_DIM_HAN"]
                
                path = self.cfgs["PATH_TO_SAVE_MODEL_HAN"]
                #path_pt = self.cfgs["PATH_TO_SAVE_MODEL_HAN_pt"]
                LAST_SAVED_EPOCH_MODEL = self.cfgs["LAST_SAVED_EPOCH_HAN_MODEL"]

                model = HierarchicalAttentionNetwork(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix, device).to(device)
                
            elif self.args.MODEL == "LSTM":

                # create dataset loader for train and evaluation set.
                dataset_train = CommentDataset_LSTM(data_folder, train=True, test=False, valid=False)
                dataset_valid = CommentDataset_LSTM(data_folder, train=False, test=False, valid=True)

                if self.cfgs["BATCH_SAMPLER"] == False:
                    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                                shuffle=False, collate_fn=collate_batch_LSTM)
                    val_dataloader = DataLoader(dataset_valid,
                                                shuffle=False, collate_fn=collate_batch_LSTM)
                elif self.cfgs["BATCH_SAMPLER"] == True:
                    num_of_liberals = int(BATCH_SIZE / 2)
                    num_of_conservatives = int(BATCH_SIZE - num_of_liberals)

                    batch_sampler_train = YoutubeBatchSampler(dataset_train, num_of_liberals, num_of_conservatives)

                    train_dataloader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                                collate_fn=collate_batch_LSTM)
                    val_dataloader = DataLoader(dataset_valid,
                                                shuffle=False, collate_fn=collate_batch_LSTM)
                
                # create/load vocabulary
                
                PATH_TO_VOCABULARY = self.cfgs["PATH_TO_SAVE_VOCAB_LSTM"]
                vocab, vocab_size = get_vocabulary(PATH_TO_VOCABULARY, "LSTM", dataset_train)

                embedding_matrix = get_embedding_matrix(USE_PRETRAINED_EMBEDDING_MATRIX, vocab)

                HIDDEN_DIMENSION = self.cfgs["HIDDEN_DIM_LSTM"]

                path = self.cfgs["PATH_TO_SAVE_MODEL_LSTM"]
                #path_pt = self.cfgs["PATH_TO_SAVE_MODEL_HAN_pt"]
                LAST_SAVED_EPOCH_MODEL = self.cfgs["LAST_SAVED_EPOCH_LSTM_MODEL"]

                model = UserClassificationModel(vocab_size, embedding_size, num_class, HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix).to(device)
                
            optimizer = Adam(model.parameters(), learning_rate)
            # to train and save/load the model
            train_loss_list, val_loss_list, train_accu_list, val_accu_list, current_epoch = train(model, train_dataloader, val_dataloader, dataset_train, dataset_valid, 
                                                                                                EPOCHS, optimizer, path, LAST_SAVED_EPOCH_MODEL, device)

            plot(train_loss_list, val_loss_list, train_accu_list, val_accu_list, current_epoch, self.args.MODEL)

        elif self.args.RUN_MODE == "test":
            
            
            if self.args.MODEL == "HAN":
                dataset_test = CommentDataset(data_folder, train=False, test=True, valid=False)
                test_dataloader = DataLoader(dataset_test,
                                            shuffle=False, collate_fn=collate_batch)

                # create/load vocabulary
                PATH_TO_VOCABULARY = self.cfgs["PATH_TO_SAVE_VOCAB_HAN"]
                if PATH_TO_VOCABULARY is None:
                    raise ValueError("No Vocabulary Found. No HAN Model Trained! Train some models using `python run.py --MODEL HAN --RUN_MODE train` ")
                
                vocab, vocab_size = get_vocabulary(PATH_TO_VOCABULARY, "HAN", None)

                embedding_matrix = get_embedding_matrix(USE_PRETRAINED_EMBEDDING_MATRIX, vocab)

                ENCODER_HIDDEN_DIMENSION = self.cfgs["ENCODER_HIDDEN_DIM_HAN"]
                DECODER_HIDDEN_DIMENSION = self.cfgs["DECODER_HIDDEN_DIM_HAN"]

                path = self.cfgs["PATH_TO_SAVE_MODEL_HAN"]
                #path_pt = self.cfgs["PATH_TO_SAVE_MODEL_HAN_pt"]
                LAST_SAVED_EPOCH_MODEL = self.cfgs["LAST_SAVED_EPOCH_HAN_MODEL"]
                if LAST_SAVED_EPOCH_MODEL is None:
                    raise ValueError("No HAN Model Trained! Train some models using `python run.py --MODEL HAN --RUN_MODE train` ")

                model = HierarchicalAttentionNetwork(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix, device).to(device)
                
            elif self.args.MODEL == "LSTM":
                dataset_test = CommentDataset_LSTM(data_folder, train=False, test=True, valid=False)
                test_dataloader = DataLoader(dataset_test,
                                            shuffle=False, collate_fn=collate_batch_LSTM)

                # create/load vocabulary
                PATH_TO_VOCABULARY = self.cfgs["PATH_TO_SAVE_VOCAB_LSTM"]
                if PATH_TO_VOCABULARY is None:
                    raise ValueError("No Vocabulary Found. No LSTM Model Trained! Train some models using `python run.py --MODEL LSTM --RUN_MODE train` ")
                
                vocab, vocab_size = get_vocabulary(PATH_TO_VOCABULARY, "LSTM", None)

                embedding_matrix = get_embedding_matrix(USE_PRETRAINED_EMBEDDING_MATRIX, vocab)

                HIDDEN_DIMENSION = self.cfgs["HIDDEN_DIM_LSTM"]

                path = self.cfgs["PATH_TO_SAVE_MODEL_LSTM"]
                LAST_SAVED_EPOCH_MODEL = self.cfgs["LAST_SAVED_EPOCH_LSTM_MODEL"]
                if LAST_SAVED_EPOCH_MODEL is None:
                    raise ValueError("No LSTM Model Trained! Train some models using `python run.py --MODEL LSTM --RUN_MODE train` ")

                model = UserClassificationModel(vocab_size, embedding_size, num_class, HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix).to(device)
                
            # to load the model:
            model, _, _, _, _, _, _ = load_model(path, LAST_SAVED_EPOCH_MODEL, model, None)

            loss_function = nn.CrossEntropyLoss()
            TEST_ACC, TEST_LOSS, F1_score = evaluate(test_dataloader, model, dataset_test, loss_function, device)
            print("The test accuracy is: {:.2f}%".format(TEST_ACC))
            print("F1 Score on Test data is: {:.2f}".format(F1_score))
            print("Loss on Test Data is: {:.2f}".format(TEST_LOSS))

if __name__ == "__main__":
    args = parse_args()

    # download and extract the data. This will run only once. 
    #if((os.path.isfile(os.path.join('Data', 'europarl-v7.es-en.en')) == False) or (os.path.isfile(os.path.join('Data', 'europarl-v7.es-en.es')) == False) ):
    #    Download_and_extract()
    
    with open('./config.yml', 'r') as f:
        model_config = yaml.safe_load(f)

    exec = MainExec(args, model_config)
    exec.run(args)