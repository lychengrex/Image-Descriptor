import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import os
import glob
import pickle
from preprop import CocoDataset, collate_fn
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


class Args():
    def __init__(self):
        '''
        For jupyter notebook
        '''
        self.model_path = 'models/'
        self.vocab_path = 'data/vocab.pkl'
        self.image_dir = 'data/resized2014'
        self.caption_path = 'data/annotations/captions_train2014.json'
        self.log_step = 10
        self.save_step = 1000
        self.embed_size = 256
        self.hidden_size = 512
        self.num_layers = 1
        self.num_epochs = 5
        self.batch_size = 128
        self.num_workers = 2
        self.learning_rate = 0.001

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    return parser.parse_args()

def main(args):

    dataset = CocoDataset(root=args.image_dir,
                                json=args.caption_path, vocab=self.__vocab)

    exp = Experiement(args)
    exp.run()


if __name__ == '__main__':
    
    args = parser.parse_args()
    print(args)
    main(args)
