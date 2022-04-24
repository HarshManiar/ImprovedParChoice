import argparse, sys, torch, random, pickle, time
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from os import path, remove
from main_parchoice import parchoice
from inference import inference
from context_preservation import preserve_context

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

def parchoice_only(src_test, tgt_test, src_train, tgt_train, output_src='parchoice_only_out_src.txt', output_tgt='parchoice_only_out_tgt.txt', verbose=False):
    src_transformed = parchoice(src_test, src_train, tgt_train, verbose=verbose)
    tgt_transformed = parchoice(tgt_test, tgt_train, src_train, verbose=verbose, flip_tgt=True)
    src_transformed = [line.replace('@ ', '@') for line in src_transformed]
    tgt_transformed = [line.replace('@ ', '@') for line in tgt_transformed]
    with open('tmp_pc_src.txt', 'w') as file:
        for line in src_transformed:
            file.write(line + '\n')
    with open('tmp_pc_tgt.txt', 'w') as file:
        for line in tgt_transformed:
            file.write(line + '\n')
    preserve_context(src_test, 'tmp_pc_src.txt', src_train, tgt_train, output=output_src)
    preserve_context(tgt_test, 'tmp_pc_tgt.txt', tgt_train, src_train, output=output_tgt)
    remove('tmp_pc_src.txt')
    remove('tmp_pc_tgt.txt')

def transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, output_src='transformer_only_out_src.txt', output_tgt='transformer_only_out_tgt.txt', verbose=False):
    config = Config()
    tgt_to_src_out, src_to_tgt_out = inference(config, fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, verbose=verbose)
    with open('tmp_pc_src.txt', 'w') as file:
        for line in src_to_tgt_out:
            file.write(line + '\n')
    with open('tmp_pc_tgt.txt', 'w') as file:
        for line in tgt_to_src_out:
            file.write(line + '\n')
    preserve_context(src_test, 'tmp_pc_src.txt', src_train, tgt_train, output=output_src)
    preserve_context(tgt_test, 'tmp_pc_tgt.txt', src_train, tgt_train, output=output_tgt)
    remove('tmp_pc_src.txt')
    remove('tmp_pc_tgt.txt')

def serial_parchoice_transformer(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, verbose=False):
    if not path.exists('parchoice_only_out_src.txt') or not path.exists('parchoice_only_out_tgt.txt'):
        parchoice_only(src_test, tgt_test, src_train, tgt_train, output_src='tmp_pc_only_src.txt', output_tgt='tmp_pc_only_tgt.txt', verbose=verbose)
        transformer_only(fpath, dpath, src_train, src_dev, 'tmp_pc_only_src.txt', tgt_train, tgt_dev, 'tmp_pc_only_tgt.txt', output_src='serial_parchoice_transformer_out_src.txt', output_tgt='serial_parchoice_transformer_out_tgt.txt', verbose=verbose)
        remove('tmp_pc_only_src.txt')
        remove('tmp_pc_only_tgt.txt')
    else:
        transformer_only(fpath, dpath, src_train, src_dev, 'parchoice_only_out_src.txt', tgt_train, tgt_dev, 'parchoice_only_out_tgt.txt', output_src='serial_parchoice_transformer_out_src.txt', output_tgt='serial_parchoice_transformer_out_tgt.txt', verbose=verbose)

def serial_transformer_parchoice(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, verbose=False):
    if not path.exists('transformer_only_out_src.txt') or not path.exists('transformer_only_out_tgt.txt'):
        transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, output_src='tmp_transformer_only_src.txt', output_tgt='tmp_transformer_only_tgt.txt', verbose=verbose)
        parchoice_only('tmp_transformer_only_src.txt', 'tmp_transformer_only_tgt.txt', src_train, tgt_train, output_src='serial_transformer_parchoice_out_src.txt', output_tgt='serial_transformer_parchoice_out_tgt.txt', verbose=verbose)
        remove('tmp_transformer_only_src.txt')
        remove('tmp_transformer_only_tgt.txt')
    else:
        parchoice_only('transformer_only_out_src.txt', 'transformer_only_out_tgt.txt', src_train, tgt_train, output_src='serial_transformer_parchoice_out_src.txt', output_tgt='serial_transformer_parchoice_out_tgt.txt', verbose=verbose)

def hybrid_parchoice_transformer(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, clf_addr, verbose=False):
    if not path.exists('transformer_only_out_src.txt') or not path.exists('transformer_only_out_tgt.txt'):
        transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, verbose=verbose)
    with open('transformer_only_out_src.txt', 'r') as file:
        src_transformed_transformer = file.readlines()
    with open('transformer_only_out_tgt.txt', 'r') as file:
        tgt_transformed_transformer = file.readlines()
    if not path.exists('serial_parchoice_transformer_out_src.txt') or not path.exists('serial_parchoice_transformer_out_tgt.txt'):
        serial_transformer_parchoice(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, verbose=verbose)
    with open('serial_parchoice_transformer_out_src.txt', 'r') as file:
        src_transformed_serial = file.readlines()
    with open('serial_parchoice_transformer_out_tgt.txt', 'r') as file:
        tgt_transformed_serial = file.readlines()

    clf = None

    with open(clf_addr, 'rb') as f:
        clf = pickle.load(f)

    optimal_src = []
    for line_transformer, line_serial in zip(src_transformed_transformer, src_transformed_serial):
        src_acc_src_transformer = clf.accuracy([line_transformer], [1])
        src_acc_src_serial = clf.accuracy([line_serial], [1])
        if src_acc_src_serial >= src_acc_src_transformer:
            optimal_src.append(line_serial)
        else:
            optimal_src.append(line_transformer)

    optimal_tgt = []
    for line_transformer, line_serial in zip(tgt_transformed_transformer, tgt_transformed_serial):
        tgt_acc_tgt_transformer = clf.accuracy([line_transformer], [0])
        tgt_acc_tgt_serial = clf.accuracy([line_serial], [0])
        if tgt_acc_tgt_serial >= tgt_acc_tgt_transformer:
            optimal_tgt.append(line_serial)
        else:
            optimal_tgt.append(line_transformer)

    if verbose:
        print("Classifier accuracy source to target (transformer only):", clf.accuracy(src_transformed_transformer, [0 for i in range(len(src_transformed_transformer))]))
        print("Classifier accuracy source to target (serial):", clf.accuracy(src_transformed_serial, [0 for i in range(len(src_transformed_serial))]))
        print("Classifier accuracy source to target hybrid:", clf.accuracy(optimal_src, [0 for i in range(len(optimal_src))]))
        print("Classifier accuracy target to source transformer only:", clf.accuracy(tgt_transformed_transformer, [1 for i in range(len(tgt_transformed_transformer))]))
        print("Classifier accuracy target to source serial:", clf.accuracy(tgt_transformed_serial, [1 for i in range(len(tgt_transformed_serial))]))
        print("Classifier accuracy target to source hybrid:", clf.accuracy(optimal_tgt, [1 for i in range(len(optimal_tgt))]))

    with open('hybrid_transformer_parchoice_out_src.txt', 'w') as file:
        for line in optimal_src:
            file.write(line)
    with open('hybrid_transformer_parchoice_out_tgt.txt', 'w') as file:
        for line in optimal_tgt:
            file.write(line)

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
    parser.add_argument('-d', '--dpath', metavar='', help='Path to saved model_D', default=None)

    parser.add_argument('-c', '--clf', metavar='', help='Path to saved classifier')

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
    if (isinstance(args.clf, str) and not path.exists(args.clf)) or args.clf is None:
        raise Exception('Error: Invalid CLF File Path')

    begin = time.time()
    print("Running tests..")
    print("")

    start = time.time()
    print("Running Parchoice Only Pipeline...")
    parchoice_only(args.src_test, args.tgt_test, args.src_train, args.tgt_train)
    elapsed = time.time()-start
    print(f"Parchoice Only Pipeline Complete. Finished in {elapsed} seconds.")
    print("")
    start = time.time()
    print("Running Transformer Only Pipeline...")
    transformer_only(args.fpath, args.dpath, args.src_train, args.src_dev, args.src_test, args.tgt_train, args.tgt_dev, args.tgt_test)
    elapsed = time.time()-start
    print(f"Transformer Only Pipeline Complete. Finished in {elapsed} seconds.")
    print("")
    start = time.time()
    print("Running Serial Parchoice Transformer Pipeline...")
    serial_parchoice_transformer(args.fpath, args.dpath, args.src_train, args.src_dev, args.src_test, args.tgt_train, args.tgt_dev, args.tgt_test)
    elapsed = time.time()-start
    print(f"Serial Parchoice Transformer Pipeline Complete. Finished in {elapsed} seconds.")
    print("")
    start = time.time()
    print("Running Serial Transformer Parchoice Pipeline...")
    serial_transformer_parchoice(args.fpath, args.dpath, args.src_train, args.src_dev, args.src_test, args.tgt_train, args.tgt_dev, args.tgt_test)
    elapsed = time.time()-start
    print(f"Serial Transformer Parchoice Pipeline Complete. Finished in {elapsed} seconds.")
    print("")
    start = time.time()
    print("Running Hybrid Pipeline...")
    hybrid_parchoice_transformer(args.fpath, args.dpath, args.src_train, args.src_dev, args.src_test, args.tgt_train, args.tgt_dev, args.tgt_test, args.clf)
    elapsed = time.time()-start
    print(f"Hybrid Pipeline Complete. Finished in {elapsed} seconds.")
    print("")
    print(f"Testing complete. Elapsed time: {time.time()-begin}")