import os
import argparse
import logging

import datetime


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-dir', type=str, default='./data')

    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    parser.add_argument('--model-dir', type=str, default='./model/'+dt)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--nodes', type=str, default='features.csv')
    parser.add_argument('--target_ntype', type=str, default='TransactionID')
    parser.add_argument('--edges', type=str, default='relation*')
    parser.add_argument('--labels', type=str, default='tags.csv')
    parser.add_argument('--new_accounts', type=str, default='test.csv')
    parser.add_argument('--compute-metrics', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='compute evaluation metrics after training')
    parser.add_argument('--threshold', type=float, default=0, help='threshold for making predictions, default : argmax')
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_hidden', type=int, default=16, help='number of hidden units') 
    parser.add_argument('--n_layers', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight for L2 loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability, for gat only features')
    parser.add_argument('--embedding-size', type=int, default=360, help="embedding size for node embedding")
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha parameter for LeakyReLU')

    return parser.parse_known_args()[0]


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger