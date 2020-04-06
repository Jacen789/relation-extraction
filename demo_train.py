import os

from relation_extraction.train import train
from relation_extraction.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    train(hparams)


if __name__ == '__main__':
    main()
