import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam

from Evaluate import evaluate
from model_utils import save_model, load_model

#from run import device

def train(model, train_dataloader, val_dataloader, dataset_train, dataset_valid, EPOCHS, learning_rate, 
        path, LAST_SAVED_EPOCH_MODEL, device):
    optimizer = Adam(model.parameters(), learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # load the model if there are already saved model
    if LAST_SAVED_EPOCH_MODEL is not None:
        print("Loading model from previous iterations")
        model, optimizer, epoch_, train_loss_list, val_loss_list, train_accu_list, val_accu_list = load_model(path, LAST_SAVED_EPOCH_MODEL, model, optimizer)
        epoch_ += 1
    else:
        epoch_ = 0
        train_loss_list = []
        val_loss_list = []
        train_accu_list = []
        val_accu_list = []

    for epoch in range(epoch_, EPOCHS):
        print(epoch)
        model.train()
        model.to(device)
        train_loss = 0
        
        accuracy = 0
        
        for idx, (text, label) in enumerate(train_dataloader):
            text = text.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            predicted_label = model(text)

            loss = loss_function(predicted_label, label)

            loss.backward()
            optimizer.step()
            
            #prediction = predicted_label#.argmax(1)#.item()
            actual = label.reshape(-1)
            
            predicted_label = torch.argmax(predicted_label, dim=1 ) 
            accuracy += torch.eq(predicted_label, actual).sum().item()

            train_loss += loss.item()
        train_loss = train_loss / len(train_dataloader)
        accuracy = accuracy * 100.0 / len(dataset_train)

        EPOCH_VAL_ACC, EPOCH_VAL_LOSS, F1_score = evaluate(val_dataloader, model, dataset_valid, loss_function, device)

        print(f'Epoch: {epoch+1} | Train Loss: {train_loss} | Accuracy: {accuracy} | Val Accuracy: {EPOCH_VAL_ACC} | Val Loss: {EPOCH_VAL_LOSS} | F1 Score: {F1_score}')
        train_loss_list.append(train_loss)
        val_loss_list.append(EPOCH_VAL_LOSS)
        train_accu_list.append(accuracy)
        val_accu_list.append(EPOCH_VAL_ACC)

        # save the model.
        save_model(epoch, model, optimizer, train_loss_list, val_loss_list, train_accu_list, val_accu_list, path)
        
    return train_loss_list, val_loss_list, train_accu_list, val_accu_list, epoch+1