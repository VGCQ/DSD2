import torch
from tqdm import tqdm 

def eval(model,test_loader, loss_fn, device):
    model.eval()
    
    running_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs,labels=inputs.to(device), labels.to(device)
                outputs=model(inputs)

                loss= loss_fn(outputs,labels)
                running_loss+=loss.item()
        
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
    test_loss=running_loss/len(test_loader)
    test_accu=100.*correct/total
    
    return(test_accu, test_loss)
