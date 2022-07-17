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
import os

from sklearn.utils import shuffle

#from utils import DIRECTORY_PATH
DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"


tokenizer = get_tokenizer('basic_english')

from torch.utils.data import Dataset, DataLoader, random_split
class CommentDataset(Dataset):
    def __init__(self, folder_path):    
      
        # Read the dataset
        self.data = pd.read_csv(folder_path, sep = ",")#.head(20)
        
        self.data = shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)
        
        self.data['Annot'].replace('LEFT', 0, inplace=True)
        self.data['Annot'].replace('RIGHT', 1, inplace=True)
        
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
        
        comments = []
        for com in self.data["comment"]:
            comments.append(com.split("-|-")[:-1])
        self.comment = comments
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
        #if self.train == True:
        #    self.data_selected = self.train_df

        #elif self.test == True:
        #    self.data_selected = self.test_df

        #elif self.valid == True:
        #    self.data_selected = self.valid_df

        label = self.data.iloc[idx]["Annot"]
        sentence = self.comment[idx]
        return sentence, label

    def __len__(self):
        #if self.train == True:
        len_ = len(self.comment)

        #elif self.test == True:
        #    len_ = len(self.test_df)

        #elif self.valid == True:
        #    len_ = len(self.valid_df)
        return len_

dataset_train = CommentDataset(os.path.join(DIRECTORY_PATH, "data/12. Subscription Training Data.csv"))

def yield_tokens(data_iter):
    for iter_, _ in data_iter:
        for sentence in iter_:
            yield tokenizer(sentence)

# create vocabulary from the training data.
vocab = build_vocab_from_iterator(yield_tokens(dataset_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

vocab_size = len(vocab.get_itos()) # len(vocab.get_stoi()) - length of the vocabulary

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# works perfectly. backup
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        
        label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.int64 ))
        
        texts = []
        for t in _text:
            texts.append(torch.tensor(text_pipeline(t), dtype=torch.int64))
        text_list.append(texts)
    
    sentence_length, word_length = get_max_length(text_list)
    
    # get the maximum length of the sentence from every batch
    #max_len_sent=0
    #for t in text_list:
    #    for t_i in t:
    #        if t_i.shape[0] > max_len_sent:
    #            max_len_sent = t_i.shape[0]
    
    text_list_p = []
    for t in text_list:
        # input shape: a list of tensors with unequal length of sentences.
        # padding to the highest length of the sequence.
        p = [ torch.cat((batch, torch.LongTensor([vocab_size-1]).repeat(word_length - len(batch))), dim=0) 
                if((word_length - batch.shape[0]) !=  0 ) else batch for batch in t]
        
        # input shape: a list of tensors with unequal length of documents.
        # padding to the highest length of the document.
        if(sentence_length - len(p)) !=  0:
            extended_sentences = [torch.LongTensor([vocab_size-1 for _ in range(word_length)] )
                                  for _ in range(sentence_length - len(p))]
            p.extend(extended_sentences)

            #p = pad_sequence(text_list[0], batch_first=False, padding_value = vocab_size-1)
            #  OUTPUT shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES] => [57, 5]
        
        p = torch.stack(p)
        # OUTPUT shape: [NUM_SENTENCES X MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH] => [5,57]
        text_list_p.append(p) # for every batch
    
    text_list_p = torch.stack(text_list_p)
    # OUTPUT shape: [BATCH_SIZE X NUM_SENTENCES X MAX_LENGTH_OF_THE_SENTENCE_IN_DOCUMENT ] => [3, 5, 57]

    #text_list_p = torch.permute(text_list_p, (2, 1, 0))
    # NOt sure, whether it should be this: OUTPUT shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X NUM_SENTENCES X BATCH_SIZE] => [57, 5, 2]
    
    # convert a list of tensors to tensors.
    # input : a list of tensors of len BATCH_SIZE
    label_list = torch.stack(label_list)   
    # OUTPUT shape: [BATCH_SIZE]
    
    return text_list_p.to(device), label_list.to(device)

# get the maximum number of sentences in a document, and maximum number of words in sentences.
def get_max_length(doc):
    """
    doc = [
        [
                [1,2,3,4,5],
               [1,2,3,4],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5]
        ], 
        [
                [1,2],
               [1,2,3,4,5,6,7,8,9],
               [1,2,3,4,5],
               [1,2,3,4],
                [1, 2,3,4,5,6]
        ]
    ]

    #sentence_in_doc, word_in_sentence = get_max_length(doc)
    sentence_in_doc -> 5, and word_in_sentence -> 9
    """
    
    sent_length_list = []
    word_length_list = []

    for sent in doc:
        sent_length_list.append(len(sent))

        for word in sent:
            word_length_list.append(len(word))

    sorted_word_length = sorted(word_length_list)
    sorted_sent_length = sorted(sent_length_list)
    
    #return sorted_sent_length[int(0.8*len(sorted_sent_length))], sorted_word_length[int(0.8*len(sorted_word_length))]
    return sorted_sent_length[-1], sorted_word_length[-1]

train_dataloader = DataLoader(dataset_train, batch_size=32,
                              shuffle=False, collate_fn=collate_batch)

class Encoder(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embed_dim = EMBEDDING_DIMENSION
        
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIMENSION
        self.decoder_hidden_dim = DECODER_HIDDEN_DIMENSION
        
        self.num_class = num_class
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.gru = nn.GRU(self.embed_dim, self.encoder_hidden_dim, bidirectional =True)

        self.attention = Attention(self.encoder_hidden_dim*2)
        
    def forward(self, text):
        
        # input shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X NUM_SENTENCES X BATCH_SIZE] 
        # or input shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X NUM_SENTENCES] 
        embedded = self.embedding(text)
        # output shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X NUM_SENTENCES X EMBEDDING_DIMENSION]
        # 2nd output: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC X 100]
        
        # input shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION]
        gru_out, hidden = self.gru(embedded)
        # gru_out shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        # hidden[0] shape: [1, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION]
        # hidden[1] shape: [1, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION]
        
        # 2nd gru_out shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, ENCODER_HIDDEN_DIMENSION*2]
        # 2nd hidden shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, ENCODER_HIDDEN_DIMENSION]
        
        # concatenate both forward and backward hidden vectors
        #hidden_f_b = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim = 1)
        # output shape: [NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        

        s_i = self.attention(gru_out) # fome the diagram, it is s_i.
        
        return s_i, gru_out

class Sentence_Encoder(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embed_dim = EMBEDDING_DIMENSION
        
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIMENSION
        self.decoder_hidden_dim = DECODER_HIDDEN_DIMENSION
        
        self.num_class = num_class
        
        #self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.gru = nn.GRU(self.encoder_hidden_dim*2, self.encoder_hidden_dim, bidirectional =True)

        self.attention = Attention(self.encoder_hidden_dim*2)

        #self.init_weights()
        
    def forward(self, word_embed):
        # input shape: [BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION*2]
        gru_out, hidden = self.gru(word_embed)
        # gru_out shape: [BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION*2]
        # hidden shape: [BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION]
        
        # concatenate both forward and backward hidden vectors
        #hidden_f_b = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim = 1)
        # output shape: [NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        
        v = self.attention(gru_out) # from the diagram, it is v.
        # output: [BATCH X 1 X ENCODER_HIDDEN_DIMENSION*2]
        return v, gru_out
    
    
class Attention(nn.Module):
    def __init__(self, ENCODER_HIDDEN_DIMENSION):
        super().__init__()
        
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIMENSION
        
        self.linear = nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim)
        self.context = nn.Linear(self.encoder_hidden_dim, 1)
        
    def forward(self, gru_out):
        
        # input: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        hidden_enc = self.linear(gru_out)
        # output: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]

        # 2nd output shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, ENCODER_HIDDEN_DIMENSION*2]
    
        
        # input: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        u = torch.tanh(hidden_enc)
        # output: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        
        # 2nd output shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, ENCODER_HIDDEN_DIMENSION*2]
        
        # input: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        context_vector = self.context(u)
        # output: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, 1]
        
        # 2nd output shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, 1]
        
        # input: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, 1]
        alpha = F.softmax(context_vector, dim=1)
        # output: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, 1]
        
        # 2nd output shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, 1]
        
        alpha=alpha.permute(0, 2, 1)
        # 2nd output shape: [BATCH_SIZE, 1, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC]
        
        a = alpha@gru_out  
        # 2nd output shape: [BATCH_SIZE, 1, ENCODER_HIDDEN_DIMENSION*2]
        return a
    
class HierarchicalAttentionNetwork(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embedding_size = EMBEDDING_DIMENSION
        self.num_class = num_class
        self.ENCODER_HIDDEN_DIMENSION = ENCODER_HIDDEN_DIMENSION
        self.DECODER_HIDDEN_DIMENSION = DECODER_HIDDEN_DIMENSION

        self.model = Encoder(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION).to(device)
        self.sent_model = Sentence_Encoder(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION).to(device)
        
        self.linear = nn.Linear(self.ENCODER_HIDDEN_DIMENSION*2, self.num_class)
        
    def forward(self, text):
        
        text = text.permute(1, 0, 2)
        word_a_list, word_s_list = [], []

        # Iterate through all the sentences in every batch
        for sent in text:
            # input: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC]
            word_s, gru_out = self.model(sent)
            # output: word_s: [BATCH_SIZE, 1, ENCODER_HIDDEN_DIMENSION*2]

            #word_a_list.append(word_a)
            word_s_list.append(word_s)

        word_s_list = torch.cat(word_s_list, dim=1)
        # output: word_s: [BATCH_SIZE, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        
        v, gru_out_sentence = self.sent_model(word_s_list)
        # output v: # output: [BATCH X 1 X ENCODER_HIDDEN_DIMENSION*2]
        # output gru_out_sentence: [BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION*2]
        
        v_output = self.linear(v)
        # v_output shape: [BATCH, 1, Num_classes]        
        
        classifier = F.softmax(v_output, dim=2).squeeze(1)
        # classifier shape: [BATCH, 1, Num_classes]        
        
        return classifier
    
num_class = 2
embedding_size = 100
ENCODER_HIDDEN_DIMENSION = 64
DECODER_HIDDEN_DIMENSION = 32

han_model = HierarchicalAttentionNetwork(vocab_size, embedding_size, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION).to(device)

def train(han_model, train_dataloader):
    optimizer = Adam(han_model.parameters(), 0.0001)
    loss_function = nn.CrossEntropyLoss()

    train_loss_list = []
    for epoch in range(0, 20):
        han_model.train()
        han_model.to(device)
        train_loss = 0

        for idx, (text, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            predicted_label = han_model(text)

            prediction = predicted_label#.argmax(1)#.item()
            actual = label.reshape(-1)

            predited_label = torch.argmax(prediction, dim=1 ) 
            accuracy += torch.eq(predited_label, actual).sum().item()

            loss = loss_function(prediction, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        accuracy = accuracy * 100.0 / len(dataset_train)
        print(f'Epoch: {epoch+1} | Train Loss: {train_loss} | Val. Loss: ')   
        train_loss_list.append(train_loss)
        
    return train_loss_list

train_loss_list = train(han_model, train_dataloader)