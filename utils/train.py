import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):

    model.train()
    running_loss=0
    correct=0
    total=0

    for data in tqdm(train_loader):
        
        inputs,labels=data[0].to(device),data[1].to(device)
        
        outputs=model(inputs)

        loss =loss_fn(outputs,labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss=running_loss/len(train_loader)
    train_accu=100.*correct/total

    return train_accu, train_loss

def train_one_epoch_KD(student_model, teacher_model, epoch, train_loader, optimizer, crossloss_fn, T, alpha, device):
    
    student_model.train()
    teacher_model.eval()

    running_loss=0
    correct=0
    total=0
    running_loss_kl=0 
    running_loss_cross=0 

    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs,labels=inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            y_stud = student_model(inputs)
            with torch.no_grad():
                y_teach = teacher_model(inputs)

            cross_loss = crossloss_fn(y_stud, labels) 
            kl_loss = F.kl_div(F.log_softmax(y_stud / T, dim=1), F.softmax(y_teach / T, dim=1), reduction='batchmean')
            loss =  kl_loss * T * T * alpha + cross_loss * (1. - alpha)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_kl+=kl_loss 
            running_loss_cross+=cross_loss 

            _, predicted = y_stud.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    train_loss = running_loss/len(train_loader)
    train_cross_loss = running_loss_cross/len(train_loader) 
    train_kl_loss = running_loss_kl/len(train_loader) 
    train_accu = 100.*correct/total
    
    return(train_accu, train_loss, train_kl_loss, train_cross_loss)
