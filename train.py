import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import os
import argparse
import json

# First thing is to parse out our args
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network!")
    parser.add_argument('data_directory', help='Point us to your data')
    parser.add_argument('--save_dir', help='Where do you want to save your neural network', default='checkpoint.pth')
    parser.add_argument('--arch', help='models to use for neural network', choices=['vgg', 'densenet'], default='vgg')
    parser.add_argument('--epochs', default='3')
    parser.add_argument('--learning_rate', default='0.003')
    parser.add_argument('--hidden_units', default=4096)
    parser.add_argument('--gpu', default='gpu')
    return parser.parse_args()

def load_data(args):
    print("Suggested data directory {}".format(args.data_directory))
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    
    #Open up the json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])
    
    test_validate_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_validate_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_validate_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    
    return trainloader, testloader, validloader, train_data
    

def create_model(args):
    if args.arch is None or args.arch == 'vgg':
        input_node = 25088
        model = models.vgg11(pretrained=True)
    else:
        input_node = 1024
        model = models.densenet121(pretrained=True)
        
    hidden_units = 4096
    learning_rate = 0.003
    if args.hidden_units is not None:
        hidden_units = args.hidden_units
    
    if args.learning_rate is not None:
        learing_rate = float(args.learning_rate)
    
    for param in model.parameters():
        param.requires_grad = False
     
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_node, int(hidden_units))),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(learning_rate)),
        ('fc2', nn.Linear(int(hidden_units), 102)),
        ('output', nn.LogSoftmax(dim=1))]))
    return model
     

def save_model(model, args, optimizer):
    
    checkpoint = {
    'arch': args.arch,
     'model': model,
    'learning_rate': args.learning_rate,
    'output_size': args.hidden_units,
    'classifier': model.classifier,
    'features': model.features,
    'learning_rate': args.learning_rate,
    'epochs': args.epochs,
    'optimizer': optimizer,
    'optimizer_state': optimizer.state_dict(),
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    }

    torch.save(checkpoint, args.save_dir)
        

# Main Training function
def train(model, criterion, optimizer, dataloader, validloader, epochs, gpu):
    print("Starting our training now!")
    steps = 0
    check_in = 20
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloader):
            steps += 1
            
            # need to check what gpu got passed in
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                model.cpu()
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % check_in == 0:
                model.eval()
                accuracy = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for ii, (valid_image, valid_label) in enumerate(validloader):
                        if gpu == 'gpu':
                            valid_image, valid_label = valid_image.to('cuda'), valid_label.to('cuda')
                        else:
                            pass
                        valid_output = model.forward(valid_image)
                        _, predicted = torch.max(valid_output.data, 1)
                        total += valid_label.size(0)
                        correct_run = (predicted == valid_label).sum().item()
                        correct += correct_run
                        accuracy = (correct / total)
                print("Epoch: {}/{}".format(e + 1, epochs))
                print("Loss: {:.5f}".format(running_loss/check_in))
                print("Accuracy off Validation set: {}".format(round(accuracy, 5)))       
        
    print('DONE TRAINING YASHIE!')
    return model
        
def main():
    print('Starting train.py!')
    args = parse_args()
    
    trainloader, testloader, validloader, train_data = load_data(args)
    model = create_model(args)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    model = train(model, criterion, optimizer, trainloader, validloader, int(args.epochs), args.gpu)
    model.class_to_idx = train_data.class_to_idx
    save_model(model, args, optimizer)

main()   