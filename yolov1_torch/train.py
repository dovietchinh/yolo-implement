import argparse
import torch
import os
import numpy as np
import random
from tqdm import tqdm
from utils.torch_utils import select_device
def train(opt): 
    pass 

def parse_opt(know=False): 
    parse = argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default='yolov1.pt', help='init weights path')
    parse.add_argument('--cfg', type=str, default='models/hub/yolov1.yaml', help='model.yaml path')
    parse.add_argument('--data', type=str, default='coco.yaml')
    parse.add_argument('--epochs',type=int, default=300)
    parse.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUS')
    parse.add_argument('--imgsize', type=int, default=448, help='train val images size')
    parse.add_argument('--device', type=int, default=0, )
    parse.add_argument('--project', type=str, default='runs/train')
    parse.add_argument('--name', type=str, default='exp')
    opt = parser.parse_know_args()[0] parse.parse_args()
    return opt 

def train()




def main(opt): 
    device = select_device(opt.device)

def run(**kwargs):
    opt = parse_opt(True)
    for k,v in kwargs.items(): 
        setattr(opt,k,v)
    main(opt)

if __name__ =='__main__':
    opt = parse_opt()
    main(opt)
