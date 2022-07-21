import torch
import torch.nn.functional as F


from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#from run import device

def evaluate(val_dataloader, model, dataset_valid, loss_function, device, INFERENCE=None):
    model.eval()
    correct = 0
    total_count = 0

    # for f1 score
    prediction_labels = []
    actual_labels = []

    val_loss = 0

    with torch.no_grad():    
        for text, label in val_dataloader:
            text = text.to(device)
            label = label.to(device)
            
            # feed the validation text into the model, and get the probabilities.
            predicted_label, alpha_word, alpha_sentence = model(text)

            # visualizing text
            if INFERENCE ==True:
                print("inferencing")
                print(alpha_word, alpha_sentence)
            
            # calculate loss
            loss = loss_function(predicted_label, label)
            
            # validation accuracy
            actual = label.reshape(-1)
            predicted_label = torch.argmax(predicted_label, dim=1 ) 
            correct += torch.eq(predicted_label, actual).sum().item()

            # to cal f1 score.
            prediction_labels.append(predicted_label)
            actual_labels.append(actual)   

            # convert probabilities into 0/1.
            #predicted_label = torch.round(predicted_label).type(torch.int64)
            
            # count the number of correctly predicted labels.
            #correct += torch.eq(predicted_label, label).sum().item()
            
            # get the total length of the sentences in val_dataloader
            #total_count += label.size(0)

            val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)

        # convert unequal length of lists of tensors to on single tensors.
        #actual_labels = torch.flatten(torch.stack(actual_labels)) 
        actual_labels = torch.cat(actual_labels).to('cpu')
        #prediction_labels = torch.flatten(torch.stack(prediction_labels)) 
        prediction_labels = torch.cat(prediction_labels).to('cpu')

        F1_score = f1_score(actual_labels, prediction_labels)
        
    
    # returns the accuracy of the model
    return correct * 100.0 / len(dataset_valid), val_loss, F1_score