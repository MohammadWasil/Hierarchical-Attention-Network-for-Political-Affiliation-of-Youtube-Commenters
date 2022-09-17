import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import os, zipfile
import numpy as np

tokenizer = get_tokenizer('basic_english')

def load_vocabulary(path):
    global vocab, vocab_size
    checkpint_vocab = torch.load(path)

    vocab = checkpint_vocab['vocab']
    vocab_size = checkpint_vocab['vocab_size']
    return vocab, vocab_size

def save_vocabulary(path, vocab, vocab_size):
    torch.save({
                'vocab': vocab,
                'vocab_size': vocab_size
                }, path)

def yield_tokens(data_iter):
    for iter_, _ in data_iter:
        for sentence in iter_:
            yield tokenizer(sentence)

def yield_tokens_LSTM(data_iter):
    for sentence, _ in data_iter:
        yield tokenizer(sentence)

def vocabulary(dataset_train, model):
    global vocab, vocab_size
    # create vocabulary from the training data.
    if model == "HAN":
        vocab = build_vocab_from_iterator(yield_tokens(dataset_train), specials=["<unk>"])
    elif model == "LSTM":
        vocab = build_vocab_from_iterator(yield_tokens_LSTM(dataset_train), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    vocab_size = len(vocab.get_itos()) # len(vocab.get_stoi()) - length of the vocabulary
    
    return vocab, vocab_size

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)

def get_vocabulary(PATH_TO_VOCABULARY, model_name, dataset_train=None):
    if os.path.exists(PATH_TO_VOCABULARY):
        print("Loading Vocabulary ... ")
        vocab, vocab_size = load_vocabulary(PATH_TO_VOCABULARY)
    else:
        print("Creating and Saving vocabulary")
        vocab, vocab_size = vocabulary(dataset_train, model_name)
        save_vocabulary(PATH_TO_VOCABULARY, vocab, vocab_size)
    return vocab, vocab_size

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
    
    return text_list_p, label_list

def collate_batch_inference(batch):
    label_list, text_list = [], []
    author_list = []
    for _text, author in batch:
        
        #label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.int64 ))
        
        texts = []
        for t in _text:
            texts.append(torch.tensor(text_pipeline(t), dtype=torch.int64))
        text_list.append(texts)

        author_list.append(author)
    
    sentence_length, word_length = get_max_length(text_list)
    
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
        
        p = torch.stack(p)
        # OUTPUT shape: [NUM_SENTENCES X MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH] => [5,57]
        text_list_p.append(p) # for every batch
    
    text_list_p = torch.stack(text_list_p)
    # OUTPUT shape: [BATCH_SIZE X NUM_SENTENCES X MAX_LENGTH_OF_THE_SENTENCE_IN_DOCUMENT ] => [3, 5, 57]

    # convert a list of tensors to tensors.
    # input : a list of tensors of len BATCH_SIZE
    #label_list = torch.stack(label_list)   
    # OUTPUT shape: [BATCH_SIZE]
    
    return text_list_p, author_list

def collate_batch_LSTM(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.int64 ))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    
    text_list = pad_sequence(text_list, batch_first=False, padding_value = vocab_size-1)
   
    # convert a list of tensors to tensors.
    # input : a list of tensors of len BATCH_SIZE
    label_list = torch.stack(label_list)   
    # OUTPUT shape: [BATCH_SIZE]
    
    return text_list, label_list

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

def get_embedding_matrix(USE_PRETRAINED_EMBEDDING_MATRIX, vocab):
    if USE_PRETRAINED_EMBEDDING_MATRIX:
        # download an extract the glove embedding if they're not.
        load_pretrained_embedding_matrix()
        # load the embedding matrix.
        embedding_matrix = GloveModel("glove.840B.300d.txt", vocab)
    else:
        embedding_matrix = None
    
    return embedding_matrix

def Download_and_extract():
    print("This might take some time...")
    print("Downloading...")
    os.system('wget https://nlp.stanford.edu/data/glove.840B.300d.zip')
    
    Extract()
    
def Extract():
    print("Extracting...")
    # extract and save to the same directory.
    with zipfile.ZipFile('glove.840B.300d.zip', 'r') as zip_ref:
        zip_ref.extractall("./")
    print("Done!")
    
def load_pretrained_embedding_matrix():
    # Downloadin Glove word vector
    # this might take some time........... ~5 mins.
    if((os.path.isfile('glove.840B.300d.zip') == False)):
        Download_and_extract()
    elif((os.path.isfile('glove.840B.300d.zip') == True) and (os.path.isfile('glove.840B.300d.txt') == False)):
         Extract()
    else:
        print("Already Downloaded and extracted!")

    #!wget https://nlp.stanford.edu/data/glove.840B.300d.zip
    #!unzip glove.840B.300d.zip

# https://github.com/MohammadWasil/Visual-Question-Answering-VQA/blob/master/2.%20Dataset%20Used%20in%20Training..ipynb
def GloveModel(file_path, vocab):
    embedding_index = {}
    f = open(file_path,'r', encoding='utf8')
    embedding_index = {}
    print("Opened!")

    for j, line in enumerate(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        embedding_index[word] = embedding
      
    print("Done.",len(embedding_index)," words loaded!")
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(vocab.get_stoi()) + 1, EMBEDDING_DIM))
    print(embedding_matrix.shape)

    for index, word in enumerate(vocab.get_itos()):
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[index] = embedding_vector
    return embedding_matrix