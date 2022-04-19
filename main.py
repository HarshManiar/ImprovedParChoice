import argparse, sys, torch
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from os import path, remove
from main_parchoice import parchoice
from inference import inference
from context_context_preservation import preserve_context

class Config():
    data_path = './'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 64
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 32
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

    run_eval = False
    use_ref = False

def parchoice_only(src, tgt, src_train, tgt_train):
    src_transformed = parchoice(src, src_train, tgt_train)
    tgt_transformed = parchoice(tgt, src_train, tgt_train)
    with open('tmp_pc_src.txt', 'w') as file:
        for line in src_transformed:
            file.write(line + '\n')
    with open('tmp_pc_tgt.txt', 'w') as file:
        for line in tgt_transformed:
            file.write(line + '\n')
    preserve_context(src, 'tmp_pc_src.txt', src_train, tgt_train, output='parchoice_only_out_src.txt')
    preserve_context(src, 'tmp_pc_tgt.txt', src_train, tgt_train, output='parchoice_only_out_tgt.txt')
    remove('tmp_pc_src.txt')
    remove('tmp_pc_tgt.txt')

    # Perform Scoring Using Metrics Here:

def transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test):
    config = Config()
    tgt_to_src_out, src_to_tgt_out = inference(config, fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test)
    with open('tmp_pc_src.txt', 'w') as file:
        for line in src_to_tgt_out:
            file.write(line + '\n')
    with open('tmp_pc_tgt.txt', 'w') as file:
        for line in tgt_to_src_out:
            file.write(line + '\n')
    preserve_context(src_test, 'tmp_pc_src.txt', src_train, tgt_train, output='transformer_only_out_src.txt')
    preserve_context(tgt_test, 'tmp_pc_tgt.txt', src_train, tgt_train, output='transformer_only_out_tgt.txt')
    remove('tmp_pc_src.txt')
    remove('tmp_pc_tgt.txt')
    
    # Perform Scoring Using Metrics Here:

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

    