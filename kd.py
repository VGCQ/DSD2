import os 
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune
import random
import numpy as np

from utils.models import ResNet, BasicBlock, ResNet_orig, VGGlikeModel
from utils.cifar import input_dataset
from utils.train import train_one_epoch_KD
from utils.test import eval

def main():
    parser = argparse.ArgumentParser(description='DSD^2 experiments (KD)')
    parser.add_argument('--teacher_model', default='ResNet-18',  help='Teacher model for KD (default: ResNet-18)')
    parser.add_argument('--path_to_teacher_model',  help='path for the teacher model')
    parser.add_argument('--student_model', default='VGG-like',  help='student model (default: VGG-like)')
    parser.add_argument('--delta', type=int, default=5, help="depth of the NN, number of convolution block (default: 5)")
    parser.add_argument('--gamma', type=int, default=5, help="2^gamma = width of convolutional layer (default: 5)")
    parser.add_argument('--lr', type=float, help="learning rate" )
    parser.add_argument('--batch_size', type=int, help="batch size")
    parser.add_argument('--weight_decay', type=float, help="weight decay")
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--dataset', default='CIFAR-10',  help='dataset (default : CIFAR-10)')
    parser.add_argument('--data_path', help='path to dataset')
    parser.add_argument('--amount_noise', type=float, default=0.1,  help='amount of label noise (default : 0.1)')
    parser.add_argument('--alpha', type=float, default=0.8,  help='distillation hyper-parameter weighting the sum between the distillation loss and the student loss (default: 0.8)')
    parser.add_argument('--T', type=float, default=10,  help='temperature (default : 10)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='seed (default: 0)')
    parser.add_argument('--fixed_amount_pruning', type=float, default=0.2, help="percentage of weights to remove at each round (default: 0.2)")
    parser.add_argument('--device', type=int, default=0, help='GPU id (default: 0)')
    args = parser.parse_args()

    name_run = "KD_"+args.dataset+"_"+args.student_model+"distilled_from_"+args.teacher_model

    ## GPU
    cuda = "cuda:"+str(args.device)
    device = torch.device(cuda)

    ## SEEDING
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    if args.dataset == "CIFAR-10":
        num_classes = 10
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        epochs = 160
        batch_size = 128
        milestones=[80,120]
        gamma=0.1

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32,4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        train_dataset = torchvision.datasets.CIFAR10(root=args.data_path,
                                                    train=True, 
                                                    transform=transform,
                                                    download=True)

        test_dataset = torchvision.datasets.CIFAR10(root=args.data_path,
                                                    train=False, 
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                                    download=True)
        
        # Noisy Labels 
        for i in range(int(len(train_dataset)*args.amount_noise)):
            label = train_dataset.targets[i]
            a=random.randint(0,num_classes-1)
            while label == a:
                a=random.randint(0,num_classes-1)
            train_dataset.targets[i] = a
        
    elif args.dataset == "CIFAR-100":
        num_classes = 100
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        epochs = 160
        batch_size = 128
        milestones=[80,120]
        gamma=0.1

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32,4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        train_dataset = torchvision.datasets.CIFAR100(root=args.data_path,
                                                    train=True, 
                                                    transform=transform,
                                                    download=True)

        test_dataset = torchvision.datasets.CIFAR100(root=args.data_path,
                                                    train=False, 
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                                                    download=True)
        # Noisy Labels 
        for i in range(int(len(train_dataset)*args.amount_noise)):
            label = train_dataset.targets[i]
            a=random.randint(0,num_classes-1)
            while label == a:
                a=random.randint(0,num_classes-1)
            train_dataset.targets[i] = a
    
    elif args.dataset == "CIFAR-100N":
        ## Real Noise parameters :
        noise_type = "noisy100" # help : 'clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean'
        noise_path = "utils/CIFAR-100_human.pt" 
        is_human = True # Human annotations
        dataset = 'cifar100'
        noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        noise_type = noise_type_map[noise_type]

        train_cifar100_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset,test_dataset, _, _ = input_dataset(dataset, noise_type, noise_path, is_human, train_cifar100_transform, test_cifar100_transform)

        num_classes = 100
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        epochs = 160
        batch_size = 128
        milestones=[80,120]
        gamma=0.1

    elif args.dataset == "Flowers-102":
        num_classes = 102
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-5
        epochs = 160
        batch_size = 64
        milestones=[80,120]
        gamma=0.1

        transform = transforms.Compose([torchvision.transforms.Resize(156),
                                        torchvision.transforms.CenterCrop(128),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.Flowers102(args.data_path, split = 'train', transform=transform, download = True)

        test_dataset = torchvision.datasets.Flowers102(args.data_path, split = 'test', transform=transform, download = True)

    elif args.dataset == "ImageNet":
        num_classes = 1000
        learning_rate = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        epochs = 90
        batch_size = 1024
        milestones=[30,60]
        gamma=0.1
        
        train_transform = transforms.Compose([transforms.RandomResizedCrop(256),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        
        train_dataset = torchvision.datasets.ImageFolder(args.data_path, transform=train_transform)

        test_dataset = torchvision.datasets.ImageFolder(args.data_path,transform=test_transform)
    
    ## CHANGE CONFIG
    if args.lr is not None:
        learning_rate= args.lr
    if args.batch_size is not None:
        batch_size= args.batch_size
    if args.weight_decay is not None:
        weight_decay= args.weight_decay
    if args.epochs is not None:
        epochs= args.epochs 

    ## DATALOADER:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=10)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=10)
    
    ## STUDENT MODEL:
    if args.student_model == "VGG-like":
        student_model = VGGlikeModel(num_classes, args.delta, 2**(args.gamma)).to(device)
    
    elif args.student_model == "ResNet-18" and args.dataset != "ImageNet":
        student_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)

    elif args.student_model == "ResNet-18" and args.dataset == "ImageNet":
        student_model = ResNet_orig(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
    
       
    ## TEACHER MODEL:
    if args.teacher_model == "ResNet-18" and args.dataset != "ImageNet":
        teacher_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
        teacher_model = torch.load(args.path_to_teacher_model, map_location = device)

    elif args.teacher_model == "ResNet-18" and args.dataset == "ImageNet":
        teacher_model = ResNet_orig(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
        teacher_model = torch.load(args.path_to_teacher_model, map_location = device)
    
    elif args.teacher_model == "ResNet-50" and args.dataset == "ImageNet":
        teacher_model = ResNet_orig(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes).to(device)
        teacher_model = torch.load(args.path_to_teacher_model, map_location = device)

    ## LOSS/OPTIMIZER/SCHEDULER:
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    ## TRAIN DENSE STUDENT MODEL: 
    for epoch in range(0,epochs):
        train_acc, train_loss, train_kl_loss, train_cross_loss  = train_one_epoch_KD(student_model, teacher_model, epoch, train_loader, optimizer, loss_fn, args.T, args.alpha, device)
        test_acc, test_loss = eval(student_model, test_loader, loss_fn, device)
        scheduler.step()

        torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "train_acc": train_acc, "train_loss": train_loss,
                "train_kl_loss": train_kl_loss, "train_cross_loss": train_cross_loss,
                "test_acc": test_acc, "test_loss": test_loss,
                }, "checkpoints/"+name_run+"_last.pt")
    
    torch.save(student_model, "checkpoints/"+name_run)

    ## PRUNE AND RETRAIN LOOP:
    for i in range(1, 35):
        sparsity = 1-(1-args.fixed_amount_pruning)**i
        name_of_run = name_run+"_pruned_"+str(sparsity)
                
        ## GLOBAL PRUNING
        ## Prune Conv2d + Linear
        layers_to_prune = [(module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear, student_model.modules())]
        
        torch.nn.utils.prune.global_unstructured(layers_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=args.fixed_amount_pruning)
        
        ## LOSS/OPTIMIZER/SCHEDULER:
        loss_fn=torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        for epoch in range(0,epochs):
            train_acc, train_loss, train_kl_loss, train_cross_loss  = train_one_epoch_KD(student_model, teacher_model, epoch, train_loader, optimizer, loss_fn, args.T, args.alpha, device)
            test_acc, test_loss = eval(student_model, test_loader, loss_fn, device)
            scheduler.step()

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "train_acc": train_acc, "train_loss": train_loss,
                    "train_kl_loss": train_kl_loss, "train_cross_loss": train_cross_loss,
                    "test_acc": test_acc, "test_loss": test_loss,
                    }, "checkpoints/"+name_of_run+"_last.pt")
    
        torch.save(student_model, "checkpoints/"+name_of_run)
    
if __name__ == '__main__':
    main()