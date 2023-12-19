import os 
import argparse

import torch
import torchvision
from tqdm import tqdm
import random
import numpy as np
import glob

from utils.models import ResNet, BasicBlock, VGGlikeModel

## Hook function
class SaveInput():
    def __init__(self, module):
        self.inputs = []
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.inputs.append(input[0])

    def clear(self):
        self.hook.remove()

def main():
    parser = argparse.ArgumentParser(description='Entropy calculation for models trained on CIFAR-10/100')
    parser.add_argument('--model_path',help="path to the pruned models' folder")
    parser.add_argument('--dataset', default='CIFAR-10',  help='dataset (default: CIFAR-10)')
    parser.add_argument('--data_path', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size (default: 400)')
    parser.add_argument('--arch', default='VGG-like', help='Architecture (default: VGG-like)')
    parser.add_argument('--delta', type=int, default=5, help="depth of the NN, number of convolution block (default: 5)")
    parser.add_argument('--gamma', type=int, default=5, help="2^gamma = width of convolutional layer (default: 5)")
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='seed (default: 0)')
    parser.add_argument('--device', type=int, default=0, help='GPU id (default: 0)')
    args = parser.parse_args()

    ## GPU
    cuda = "cuda:"+str(args.device)
    device = torch.device(cuda)

    ## SEEDING
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    entropy_dict = {}

    for fname in glob.glob(args.model_path + '/*'):

        sparsity = float(fname.split('_')[-1]) ## To get the sparsity of pruned models
        entropy_dict[fname]={'sparsity_entropy_neurons':[]}

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        if args.dataset == "CIFAR-10":
            num_classes = 10
            train_dataset = torchvision.datasets.CIFAR10(root=args.data_path,
                                                        train=True, 
                                                        transform=transform,
                                                        download=True)
            
        elif args.dataset == "CIFAR-100":
            num_classes = 100
            train_dataset = torchvision.datasets.CIFAR100(root=args.data_path,
                                                        train=True, 
                                                        transform=transform,
                                                        download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size, 
                                                shuffle=True,
                                                num_workers=10)
        if args.arch == 'VGG-like':
            target_model = VGGlikeModel(num_classes, args.delta, args.gamma).to(device)
        elif args.arch == 'ResNet-18':
            target_model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes).to(device)
        
        target_model = torch.load(fname, map_location=device)
        target_model.eval()

        hooks = {}
        for name, module in target_model.named_modules():
            if type(module) == torch.nn.ReLU:
                hooks[name] = SaveInput(module)
        
        full_dataset_entropy=torch.zeros(len(hooks.keys()))

        for data in tqdm(train_loader):
            with torch.no_grad():
                inputs,labels=data[0].to(device),data[1].to(device)
                
                outputs=target_model(inputs)

                ## Hook filled!
                ## Calculate P_plus and P_minus
                entropy=torch.zeros(len(hooks.keys()))
                n_neurons = torch.zeros(len(hooks.keys()))
                k=0
                for key in hooks.keys():
                    # Getting the probability depending on the state (>0 or =<0)
                    p_plus = (hooks[key].inputs[0]>0).float() 
                    p_minus = 1-p_plus

                    ## Averaging on the mini-batch
                    p_plus = torch.mean(p_plus,dim=0)
                    p_minus = torch.mean(p_minus,dim=0)

                    n_neurons[k] = p_plus.shape[0]

                    estimated_entropy = -p_plus*torch.log2(torch.clamp(p_plus, min=1e-5))-p_minus*torch.log2(torch.clamp(p_minus, min=1e-5))
                    
                    entropy[k] = torch.mean(estimated_entropy)
                    k+=1

                full_dataset_entropy = full_dataset_entropy+entropy # Summing over all the training set

                ## Removing hooks to free the memory
                for key in hooks.keys():
                    hooks[key].hook.remove()

        full_dataset_entropy/=len(train_loader) # Averaging over all the training set

        entropy_dict[fname]['sparsity_entropy_neurons']=[sparsity, full_dataset_entropy, n_neurons]

        ## sparsity is the percentage of pruned parameters (already in the name of the model if saved like in main.py or kd.py)
        ## full_dataset_entropy is a list of entropy values for each activation (here ReLU layer) in the model computed on the training set
        ## n_neurons is the number of neurons per layer before the activation (useful in the case, we need a weighted-average)

    
    np.savez_compressed(args.model_path+'_entropy.npz', entropy_dict = entropy_dict)

if __name__ == '__main__':
    main()