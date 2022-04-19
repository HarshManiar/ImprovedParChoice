import argparse
import sys
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from os import path, remove
from main_parchoice import parchoice as PC
from inference import inference as INF
from context_context_preservation import preserve_context

def parchoice_only(src, src_train, tgt_train):
    src_transformed = PC.parchoice(src, src_train, tgt_train)
    with open('tmp_pc.txt', 'w') as file:
        for line in src_transformed:
            file.write(line + '\n')
    preserve_context(src, 'tmp_pc.txt', src_train, tgt_train, output='parchoice_only_out.txt')
    remove('tmp_pc.txt')

def transformer_only():
    pass

def serial_parchoice_transformer():
    pass

def serial_transformer_parchoice():
    pass

def hybrid_parchoice_transformer():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
		description='This script runs the five outlined pipelines for this project for evaluation.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-st', '--src_train', metavar='', help='Source Training File')
    parser.add_argument('-sd', '--src_dev', metavar='', help='Source Dev File')
    parser.add_argument('-sx', '--src_test', metavar='', help='Source Test File')
    parser.add_argument('-tt', '--tgt_train', metavar='', help='Target Training File')
    parser.add_argument('-td', '--tgt_dev', metavar='', help='Target Dev File')
    parser.add_argument('-tx', '--tgt_test', metavar='', help='Target Test File')
    
    parser.add_argument('-f', '--fpath', metavar='', help='Path to saved model_F')
    parser.add_argument('-d', '--dpath', metavar='', help='Path to saved model_D')

    args = parser.parse_args()

    if (isinstance(args.src_train, str) and not path.exists(args.src_train)) or args.src_train is None:
        raise Exception('Error: Invalid Source Training File Path')
    if (isinstance(args.src_dev, str) and not path.exists(args.src_dev)) or args.src_dev is None:
        raise Exception('Error: Invalid Source Dev File Path')
    if (isinstance(args.src_test, str) and not path.exists(args.src_test)) or args.src_test is None:
        raise Exception('Error: Invalid Source Test File Path')
    if (isinstance(args.tgt_train, str) and not path.exists(args.tgt_train)) or args.tgt_train is None:
        raise Exception('Error: Invalid Target Training File Path')
    if (isinstance(args.tgt_dev, str) and not path.exists(args.tgt_dev)) or args.tgt_dev is None:
        raise Exception('Error: Invalid Target Dev File Path')
    if (isinstance(args.tgt_test, str) and not path.exists(args.tgt_test)) or args.tgt_test is None:
        raise Exception('Error: Invalid Target Test File Path')
    if (isinstance(args.fpath, str) and not path.exists(args.fpath)) or args.fpath is None:
        raise Exception('Error: Invalid model_F File Path')
    if (isinstance(args.dpath, str) and not path.exists(args.dpath)) or args.dpath is None:
        raise Exception('Error: Invalid model_D File Path')

    