import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embed_dim = EMBEDDING_DIMENSION
        
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIMENSION
        self.decoder_hidden_dim = DECODER_HIDDEN_DIMENSION
        
        self.num_class = num_class

        if USE_PRETRAINED_EMBEDDING_MATRIX:
            self.vocab_size = embedding_matrix.shape[0]
            self.embed_dim = embedding_matrix.shape[1]
            
            self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embed_dim)
            self.embedding.weight=nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        else:
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
        

        alpha, s_i = self.attention(gru_out) # fome the diagram, it is s_i.
        
        return alpha, s_i, gru_out

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
        
        alpha, v = self.attention(gru_out) # from the diagram, it is v.
        # output: [BATCH X 1 X ENCODER_HIDDEN_DIMENSION*2]
        return alpha, v, gru_out
      
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
        alpha = F.softmax(context_vector, dim=1)   # this needs to be send also##################
        # output: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, NUM_SENTENCES, 1]
        
        # 2nd output shape: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC, 1]
        
        alpha=alpha.permute(0, 2, 1)
        # 2nd output shape: [BATCH_SIZE, 1, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC]
        
        a = alpha@gru_out  
        # 2nd output shape: [BATCH_SIZE, 1, ENCODER_HIDDEN_DIMENSION*2]
        return alpha, a
    
class HierarchicalAttentionNetwork(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, DECODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix, device):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embedding_size = EMBEDDING_DIMENSION
        self.num_class = num_class
        self.ENCODER_HIDDEN_DIMENSION = ENCODER_HIDDEN_DIMENSION
        self.DECODER_HIDDEN_DIMENSION = DECODER_HIDDEN_DIMENSION
        self.embedding_matrix = embedding_matrix
        self.USE_PRETRAINED_EMBEDDING_MATRIX = USE_PRETRAINED_EMBEDDING_MATRIX

        self.model = Encoder(self.vocab_size, self.embedding_size, self.num_class, self.ENCODER_HIDDEN_DIMENSION, self.DECODER_HIDDEN_DIMENSION, self.USE_PRETRAINED_EMBEDDING_MATRIX, self.embedding_matrix).to(device)
        self.sent_model = Sentence_Encoder(self.vocab_size, self.embedding_size, self.num_class, self.ENCODER_HIDDEN_DIMENSION, self.DECODER_HIDDEN_DIMENSION).to(device)
        
        self.linear = nn.Linear(self.ENCODER_HIDDEN_DIMENSION*2, self.num_class)
        
    def forward(self, text):
        
        text = text.permute(1, 0, 2)
        word_a_list, word_s_list = [], []

        # Iterate through all the sentences in every batch
        for sent in text:

            # input: [BATCH_SIZE, MAX_LENGTH_OF_THE_SENTENCE_IN_DOC]
            alpha_word, word_s, gru_out = self.model(sent)
            # output: word_s: [BATCH_SIZE, 1, ENCODER_HIDDEN_DIMENSION*2]

            word_a_list.append(alpha_word)
            word_s_list.append(word_s)

        word_s_list = torch.cat(word_s_list, dim=1)
        # output: word_s: [BATCH_SIZE, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION*2]
        
        alpha_sentence, v, gru_out_sentence = self.sent_model(word_s_list)
        # output v: # output: [BATCH X 1 X ENCODER_HIDDEN_DIMENSION*2]
        # output gru_out_sentence: [BATCH X NUM_SENTENCES x EMBEDDING_DIMENSION*2]
        
        v_output = self.linear(v)
        # v_output shape: [BATCH, 1, Num_classes]        
        
        classifier = F.softmax(v_output, dim=2).squeeze(1)
        # classifier shape: [BATCH, 1, Num_classes]        
        
        return classifier, word_a_list, alpha_sentence

class UserClassificationModel(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSION, num_class, ENCODER_HIDDEN_DIMENSION, USE_PRETRAINED_EMBEDDING_MATRIX, embedding_matrix):
        super().__init__()
        
        self.vocab_size = VOCAB_SIZE
        self.embed_dim = EMBEDDING_DIMENSION
        
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIMENSION
        
        self.num_class = num_class

        if USE_PRETRAINED_EMBEDDING_MATRIX:
            self.vocab_size = embedding_matrix.shape[0]
            self.embed_dim = embedding_matrix.shape[1]
            
            self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embed_dim)
            self.embedding.weight=nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.gru = nn.GRU(self.embed_dim, self.encoder_hidden_dim, bidirectional = True)
        
        self.fc1 = nn.Linear(self.encoder_hidden_dim*2, self.num_class)
        
    def forward(self, text):
        
        # input shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH X BATCH_SIZE] 
        embedded = self.embedding(text)
        # output shape: [MAX_SEQ_LENGTH x BATCH_SIZE x embed_dim]
        
        # input shape: [MAX_SEQ_LENGTH x BATCH_SIZE x embed_dim]
        gru_out, hidden = self.gru(embedded)
        # gru_out shape: [MAX_LENGTH_OF_THE_SENTENCE_IN_BATCH, BATCH_SIZE, ENCODER_HIDDEN_DIMENSION*2]
        # hidden[0] shape: [1, BATCH_SIZE, ENCODER_HIDDEN_DIMENSION]
        # hidden[1] shape: [1, NUM_SENTENCES, ENCODER_HIDDEN_DIMENSION]
        
        # input shape: [BATCH_SIZE, ENCODER_HIDDEN_DIMENSION*2]
        fc1 = self.fc1(gru_out[-1])
        # output shape: [BATCH_SIZE, num_classes]
        
        # input shape: [BATCH_SIZE, Num_class]
        classifier = F.softmax(fc1, dim=1)#.squeeze(1)
        # output shape: [BATCH_SIZE, Num_class]
        
        return classifier, None, None