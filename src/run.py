import argparse
import pickle
from model import ImageDescriptor
from build_vocab import Vocabulary


def main(args):
    img_descriptor = ImageDescriptor(args)
    if args.mode == 'train':
        img_descriptor.train()
    elif args.mode == 'eval':
        img_descriptor.evaluate(args.image_path, plot=args.plot)
    else:
        raise ValueError('Invalid mode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable', type=bool,
                        default=False, help='enable parser or not')
    parser.add_argument('--mode', type=str,
                        default='train', help='train or eval mode')
    parser.add_argument('--attention', type=bool,
                        default=False, help='use attention layers or not')
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_path', type=str,
                        default='png/example.png', help='image for evaluation')
    parser.add_argument('--plot', type=bool,
                        default=False, help='plot the evaluation image')
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
    args = parser.parse_args()
    print(args)
    main(args)
