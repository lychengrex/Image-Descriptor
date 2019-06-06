import argparse
import pickle
from model import DecoderRNN, ResNet, VGG
from utils import ImageDescriptor
from build_vocab import Vocabulary


def run(args):
    if args.encoder == 'resnet':
        encoder = ResNet(args.embed_size, ver=args.encoder_ver,
                         attention_mechanism=args.attention)
    elif args.encoder == 'vgg':
        encoder = VGG(args.embed_size, ver=args.encoder_ver,
                      attention_mechanism=args.attention)
    else:
        raise NameError('Not supported pretrained network')

    img_descriptor = ImageDescriptor(
        args, encoder=encoder)
    print(img_descriptor)
    if args.mode == 'train':
        # train the network
        img_descriptor.train()
    elif args.mode == 'test':
        # get image caption for one image
        img_descriptor.test(args.image_path, plot=args.plot)
    elif args.mode == 'val':
        # only run validation set for a specific epoch and get the average loss and perplexity
        img_descriptor.evaluate(print_info=True)
    else:
        raise ValueError('Invalid mode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str,
                        default='resnet', help='pretrained network for encoder')
    parser.add_argument('--encoder_ver', type=int,
                        default='101', help='number of layers of the pretrained network')
    parser.add_argument('--mode', type=str,
                        default='train', help='train, test or val mode')
    parser.add_argument('--attention', type=bool,
                        default=False, help='use attention layers or not')
    parser.add_argument('--model_dir', type=str,
                        default='../models/', help='path for saving trained models')
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='file name of the specific checkpoint(.ckpt)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='../data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_path', type=str,
                        default='../png/example.png', help='image for evaluation')
    parser.add_argument('--plot', type=bool,
                        default=False, help='plot the evaluation image')
    parser.add_argument('--image_dir', type=str,
                        default='../data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')
    parser.add_argument('--validate_when_training', type=bool, default=False,
                        help='perform validation during training or not')

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
    args = parser.parse_args()
    print(args)
    run(args)
