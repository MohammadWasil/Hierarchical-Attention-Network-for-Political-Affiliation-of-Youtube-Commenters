import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader, random_split
import re
import pandas as pd

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

tokenizer = get_tokenizer('basic_english')

data_folder = "12. Subscription Training Data.csv"

dataset_train = CommentDataset(data_folder, train=True, test=False, valid=False)
dataset_test = CommentDataset(data_folder, train=False, test=True, valid=False)
dataset_valid = CommentDataset(data_folder, train=False, test=False, valid=True)


# create vocabulary from the training data.
vocab = build_vocab_from_iterator(yield_tokens(dataset_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# create vocabulary from the training data.
vocab = build_vocab_from_iterator(yield_tokens(dataset_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(dataset_train, batch_size=32,
                              shuffle=False, collate_fn=collate_batch)
val_dataloader = DataLoader(dataset_valid,
                              shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(dataset_test,
                              shuffle=False, collate_fn=collate_batch)

# can change this accordingly.
USE_PRETRAINED_EMBEDDING_MATRIX = True

if USE_PRETRAINED_EMBEDDING_MATRIX:
    # download an dextract the glove embedding if they're not.
    load_pretrained_embedding_matrix()
    # load the embedding matrix.
    embedding_matrix = GloveModel("glove.840B.300d.txt", vocab)
else:
    embedding_matrix = None

num_class = 2
embedding_size = 100
ENCODER_HIDDEN_DIMENSION = 64
DECODER_HIDDEN_DIMENSION = 32

han_model = HierarchicalAttentionNetwork(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION).to(device)

train_loss_list, val_loss_list, train_accu_list, val_accu_list = train(han_model, train_dataloader)

loss_function = nn.CrossEntropyLoss()
TEST_ACC, TEST_LOSS, F1_score = evaluate(test_dataloader, han_model, loss_function = nn.CrossEntropyLoss())
print("The test accuracy is: {:.2f}%".format(TEST_ACC))
print("F1 Score on Test data is: {:.2f}".format(F1_score))
print("Loss on Test Data is: {:.2f}".format(TEST_LOSS))