import torch

#from datetime import datetime
def save_model(epoch, model, optimizer, train_loss_list, val_loss_list, train_accu_list, val_accu_list, path):
    # path in .pth or pt format
    #now = datetime.now()
    #current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    
    # save the file
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list,
                'train_accu_list': train_accu_list,
                'val_accu_list': val_accu_list,
                }, path.format(epoch) )

def load_model(path, last_saved_epoch, model, optimizer):

    checkpoint = torch.load(path.format(last_saved_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss_list = checkpoint['train_loss_list']
    val_loss_list = checkpoint['val_loss_list']
    train_accu_list = checkpoint['train_accu_list']
    val_accu_list = checkpoint['val_accu_list']
    
    return model, optimizer, epoch, train_loss_list, val_loss_list, train_accu_list, val_accu_list