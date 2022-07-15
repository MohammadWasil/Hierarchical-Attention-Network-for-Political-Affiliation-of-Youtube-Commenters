import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam

def train(han_model, train_dataloader):
    optimizer = Adam(han_model.parameters(), 0.0001)
    loss_function = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_accu_list = []
    val_accu_list = []

    for epoch in range(0, 20):
        han_model.train()
        han_model.to(device)
        train_loss = 0
        
        accuracy = 0
        
        for idx, (text, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            predicted_label = han_model(text)

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

        EPOCH_VAL_ACC, EPOCH_VAL_LOSS, F1_score = evaluate(val_dataloader, han_model, loss_function)

        print(f'Epoch: {epoch+1} | Train Loss: {train_loss} | Accuracy: {accuracy} | Val Accuracy: {EPOCH_VAL_ACC} | Val Loss: {EPOCH_VAL_LOSS} | F1 Score: {F1_score}')
        train_loss_list.append(train_loss)
        val_loss_list.append(EPOCH_VAL_LOSS)
        train_accu_list.append(accuracy)
        val_accu_list.append(EPOCH_VAL_ACC)
        
    return train_loss_list, val_loss_list, train_accu_list, val_accu_list