import torch
from tqdm.notebook import tqdm

def inference_data(dataloader_inference, model, device):
    model.eval()
    correct = 0
    total_count = 0
    val_loss = 0

    left = 0
    right = 0

    left_comment = 0
    right_comment = 0

    with torch.no_grad():    
        for text in tqdm(dataloader_inference):
            text = text.to(device)
            
            # feed the validation text into the model, and get the probabilities.
            predicted_label, _, _ = model(text)
            predicted_label = torch.argmax(predicted_label, dim=1 ) 

            if predicted_label == 0:
                left += 1
                left_comment += text.shape[1]
            elif predicted_label == 1:
                right += 1
                right_comment += text.shape[1]
            
    # returns the accuracy of the model
    return left, right, left_comment, right_comment