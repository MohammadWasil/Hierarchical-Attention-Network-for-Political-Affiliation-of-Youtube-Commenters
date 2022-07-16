import os
import argparse
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import HierarchicalAttentionNetwork, UserClassificationModel
from datasets import CommentDataset, YoutubeBatchSampler
from data_utils import collate_batch, load_pretrained_embedding_matrix, GloveModel, vocabulary, save_vocabulary, load_vocabulary
from Evaluate import evaluate
from Train import train
from plot import plot

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
        choices=['train', 'eval'],
        help='{train, eval}',
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
        if self.args.RUN_MODE == "train":
            print("Training")

            # create dataset loader for train and evaluation set.
            dataset_train = CommentDataset(data_folder, train=True, test=False, valid=False)
            dataset_valid = CommentDataset(data_folder, train=False, test=False, valid=True)

            # create/load vocabulary
            PATH_TO_VOCABULARY = self.cfgs["PATH_TO_SAVE_VOCAB_HAN"]
            if os.path.exists(PATH_TO_VOCABULARY):
                print("Loading Vocabulary ... ")
                vocab, vocab_size = load_vocabulary(PATH_TO_VOCABULARY)
            else:
                print("Creating and Saving vocabulary")
                vocab, vocab_size = vocabulary(dataset_train)
                save_vocabulary(PATH_TO_VOCABULARY, vocab, vocab_size)
            
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

            # can change this accordingly. on CFG
            USE_PRETRAINED_EMBEDDING_MATRIX = self.cfgs["USE_PRETRAINED_EMBEDDING_MATRIX"]

            if USE_PRETRAINED_EMBEDDING_MATRIX:
                # download an dextract the glove embedding if they're not.
                load_pretrained_embedding_matrix()
                # load the embedding matrix.
                embedding_matrix = GloveModel("glove.840B.300d.txt", vocab)
            else:
                embedding_matrix = None

            embedding_size = self.cfgs["EMBEDDING_SIZE"]
            num_class = self.cfgs["NUM_CLASS"]
            EPOCHS = self.cfgs["EPOCHS"]
            
            if self.args.MODEL == "HAN":
                ENCODER_HIDDEN_DIMENSION = self.cfgs["ENCODER_HIDDEN_DIM_HAN"]
                DECODER_HIDDEN_DIMENSION = self.cfgs["DECODER_HIDDEN_DIM_HAN"]
                
                path = self.cfgs["PATH_TO_SAVE_MODEL_HAN"]
                #path_pt = self.cfgs["PATH_TO_SAVE_MODEL_HAN_pt"]
                LAST_SAVED_EPOCH_MODEL = self.cfgs["LAST_SAVED_EPOCH_HAN_MODEL"]

                model = HierarchicalAttentionNetwork(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix, device).to(device)
            elif self.args.MODEL == "LSTM":
                HIDDEN_DIMENSION = self.cfgs["HIDDEN_DIM_LSTM"]
                model = UserClassificationModel(vocab_size, embedding_size, num_class, HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix).to(device)
            
            # to save/load the model
            train_loss_list, val_loss_list, train_accu_list, val_accu_list, current_epoch = train(model, train_dataloader, val_dataloader, dataset_train, dataset_valid, 
                                                                                                EPOCHS, learning_rate, path, LAST_SAVED_EPOCH_MODEL, device)

            plot(train_loss_list, val_loss_list, train_accu_list, val_accu_list, current_epoch)

        elif self.args.RUN_MODE == "eval":
            dataset_test = CommentDataset(data_folder, train=False, test=True, valid=False)
            test_dataloader = DataLoader(dataset_test,
                                        shuffle=False, collate_fn=collate_batch)

            loss_function = nn.CrossEntropyLoss()
            TEST_ACC, TEST_LOSS, F1_score = evaluate(test_dataloader, model, dataset_test, loss_function)
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