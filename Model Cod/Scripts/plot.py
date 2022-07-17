import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot(train_loss_list, val_loss_list, train_accu_list, val_accu_list, EPOCHS, model_name):
    fig, (ax1) = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle('Loss and accuracy for {} Model.'.format(model_name))
    epochs = EPOCHS 

    # accuracy Plot
    train_accu, = ax1[0].plot(range(1, epochs+1), train_accu_list, label="Training Accuracy")  
    val_accu, = ax1[0].plot(range(1, epochs+1), val_accu_list, label="Validation Accuracy")  

    ax1[0].legend(handles=[train_accu, val_accu])
    ax1[0].set_xlabel("Epochs")
    ax1[0].set_ylabel("Accuracy")
    ax1[0].set_title("Accuracy for every Epochs")
    ax1[0].set_xticks(range(1, epochs+1))

    train_loss, = ax1[1].plot(range(1, epochs+1), train_loss_list, label="Training Loss")  
    val_loss, = ax1[1].plot(range(1, epochs+1), val_loss_list, label="Validation Loss")  

    ax1[1].legend(handles=[train_loss, val_loss])
    ax1[1].set_xlabel("Epochs")
    ax1[1].set_ylabel("Loss")
    ax1[1].set_title("Loss for every Epochs")
    ax1[1].set_xticks(range(1, epochs+1))

    # do not need the third plot.
    #fig.delaxes(ax2[1])

    #plt.show()
    # save the plot
    fig.savefig('{}.png'.format(model_name))
