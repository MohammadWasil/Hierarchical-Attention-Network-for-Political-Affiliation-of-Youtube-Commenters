import torch
import torch.nn.functional as F

import pickle
import pandas as pd
from data_utils import text_pipeline, get_max_length

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def colorize(words, sentence, color_array):
    if words:
        cmap=matplotlib.cm.Reds
    elif sentence:
        cmap=matplotlib.cm.Blues
        words = sentence
    
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    
    return colored_string   

def visualize(word, sentence, visualization_folder):
    data = pd.read_csv(visualization_folder)
    d = data["Authors Comment"]
    leaning = data["Authors Biasness"]

    with open("Alpha word sample.pkl", "rb") as fp:
        word = pickle.load(fp)
    
    with open("Alpha Sentence sample.pkl", "rb") as fp:
        sentence = pickle.load(fp)

    s = ''
    for i in range(len(d)):
        if leaning[i] == "LEFT":
            political_leaning = "Liberal"
        elif leaning[i] == "RIGHT":
            political_leaning = "Conservative"
            
        w = d[i].split(" -|- ")[:-1]
        sen = sentence[i][0].tolist()
        
        for w_i in range(len(w)):
        
            sentences = "sentence_{}".format(w_i + 1).split()

            s += colorize(None, sentences, [sen[w_i]])
            
            color_array = word[w_i][0][0].to('cpu').tolist()
            
            s += colorize(w[w_i].split(), None, color_array[:len(w[w_i].split(" "))])
            s += '<Br/>'
        s += '<p style="color:gray">' + 'Predicted leaning: ' + political_leaning + '</p>'
        s += '<Br/>'
        s += '<Br/>'
            
    with open('Visualize sample.html', 'w') as f:
        f.write(s)

#from run import device
def read_process_data(visualization_folder, vocab_size):
    data = pd.read_csv(visualization_folder)
    
    data.sort_values(by=['Num of Comments'], ascending=False, inplace=True)
    comments = []
    for com in data["Authors Comment"]:
        comments.append(com.split("-|-")[:-1])
    text_list = []

    for (_text) in comments:        
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
    #print(text_list_p.shape)
    #print(text_list_p)    
    return text_list_p

def visualize_texts(texts, model, visualization_folder, device):
    model.eval()
    
    with torch.no_grad():    
        text = texts.to(device)
        
        # feed the validation text into the model, and get the probabilities.
        predicted_label, alpha_word, alpha_sentence = model(text)

        # visualizing text
        print("Creating Visualization ... ")
        
        with open("Alpha word sample.pkl", "wb") as fp:   #Pickling
            pickle.dump(alpha_word, fp)
        
        with open("Alpha Sentence sample.pkl", "wb") as fp:   #Pickling
            pickle.dump(alpha_sentence, fp)

        visualize(alpha_word, alpha_sentence, visualization_folder)