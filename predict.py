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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', default='5')
    parser.add_argument('--filepath', default='flowers/test/1/image_06760.jpg')
    parser.add_argument('--gpu', default='vgg')
    parser.add_argument('--category_names', default='cat_to_name.json')
    return parser.parse_args()

def load_model(args):
    checkpoint = torch.load(args.checkpoint)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    

# taken from p1
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # step 1: Open image and figure out which side is shorter
    img = Image.open(image_path)
    width, height = img.size
    current_coords = [width, height]
    new_coords = [0,0]
    
    if width > height:
        aspect_ratio = width/height
        new_coords = [int(256 * aspect_ratio), 256]
    else:
        aspect_ratio = height/width
        new_coords = [256, int(256 * aspect_ratio)]
    
    img = img.resize(new_coords)
    
    new_width, new_height = img.size
    print(new_width, new_height)
    
    l = (256 - 224)/2
    t = (256 - 224)/2
    r = (256 + 224)/2
    b = (256 + 224)/2
    
    cropped = img.crop((l, t, r, b))
    np_image = np.array(cropped)
    np_image = np_image.astype('float64')
    np_image = np_image/255
    
    #subtract the means and then divide by stdev
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    # re order color dimension to be first
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

# taken from p1
def predict(image_path, model, topk=5, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    #put it back to pytorch from numpy
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    image = image.float()
    
    if gpu == 'gpu':
        image = image.cuda()
        image = model.cuda()
        with torch.no_grad():
            outputs = model.forward(image.cuda())
    else:
        image = image.cpu()
        model = model.cpu()
        with torch.no_grad():
            outputs = model.forward(image)
    #grabbing top 5 highest probable matches
    probs, classes = torch.exp(outputs).topk(topk)

    print(probs)
    print(classes)

    #add one to the classes cause indexes here start at 0
    return probs[0].tolist(), classes[0].add(1).tolist()

def main():
    print('STARTING PREDICT.PY!')
    args = parse_args()
    model = load_model(args)
    
    #Open up the json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict(args.filepath, model, int(args.top_k,), args.gpu)
    
    labels = [cat_to_name[str(cls)] for cls in classes]
    
    print(probs)
    print(classes)
    print(labels)
    
    np_probs = np.array(probs)
    
    idx = 0
    while idx < len(labels):    
        print("Rank {}: {} with probability ".format(idx + 1, labels[idx]), "%2f" % np_probs[idx])
        idx += 1
    
    

main()
    