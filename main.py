import argparse, sys, torch, random
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from os import path, remove
from main_parchoice import parchoice
from inference import inference
from context_context_preservation import preserve_context
from surrogate_classifier import surrogate_kwargs
from style_transformation import load_inflections, load_parser, load_ppdb, load_symspell, transform, CountVectorizer, TfidfVectorizer, LogisticRegressionSurrogate, MLPSurrogate

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

def parchoice_only(src_test, tgt_test, src_train, tgt_train, output_src='parchoice_only_out_src.txt', output_tgt='parchoice_only_out_tgt.txt'):
    src_transformed = parchoice(src_test, src_train, tgt_train)
    tgt_transformed = parchoice(tgt_test, src_train, tgt_train)
    with open('tmp_pc_src.txt', 'w') as file:
        for line in src_transformed:
            file.write(line + '\n')
    with open('tmp_pc_tgt.txt', 'w') as file:
        for line in tgt_transformed:
            file.write(line + '\n')
    preserve_context(src_test, 'tmp_pc_src.txt', src_train, tgt_train, output=output_src)
    preserve_context(tgt_test, 'tmp_pc_tgt.txt', src_train, tgt_train, output=output_tgt)
    remove('tmp_pc_src.txt')
    remove('tmp_pc_tgt.txt')

    # Perform Scoring Using Metrics Here:

def transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, output_src='transformer_only_out_src.txt', output_tgt='transformer_only_out_tgt.txt'):
    config = Config()
    tgt_to_src_out, src_to_tgt_out = inference(config, fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test)
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
    
    # Perform Scoring Using Metrics Here:

def serial_parchoice_transformer(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test):
    parchoice_only(src_test, tgt_test, src_train, tgt_train, output_src='tmp_pc_only_src.txt', output_tgt='tmp_pc_only_tgt.txt')
    transformer_only(fpath, dpath, src_train, src_dev, 'tmp_pc_only_src.txt', tgt_train, tgt_dev, 'tmp_pc_only_tgt.txt', output_src='serial_parchoice_transformer_out_src.txt', output_tgt='serial_parchoice_transformer_out_tgt.txt')
    remove('tmp_pc_only_src.txt')
    remove('tmp_pc_only_tgt.txt')
    
    # Perform Scoring Using Metrics Here:

def serial_transformer_parchoice(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test):
    transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, output_src='tmp_tranformer_only_src.txt', output_tgt='tmp_transformer_only_tgt.txt')
    parchoice_only('tmp_tranformer_only_src.txt', 'tmp_transformer_only_tgt.txt', src_train, tgt_train, output_src='serial_transformer_parchoice_out_src.txt', output_tgt='serial_transformer_parchoice_out_tgt.txt')
    remove('tmp_tranformer_only_src.txt')
    remove('tmp_transformer_only_tgt.txt')
    
    # Perform Scoring Using Metrics Here:

def hybrid_parchoice_transformer(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test, clf_type='lr', clf_vectorizer='count', clf_feat='word', clf_ngram_range=(1,1), clf_max_feats=10000):
    if not path.exists('transformer_only_out_src.txt') or not path.exists('transformer_only_out_tgt.txt'):
        transformer_only(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test)
    if not path.exists('serial_parchoice_transformer_out_src.txt') or not path.exists('serial_parchoice_transformer_out_tgt.txt'):
        serial_transformer_parchoice(fpath, dpath, src_train, src_dev, src_test, tgt_train, tgt_dev, tgt_test)

    src = open(src_test, 'r').readlines()
    tgt = open(tgt_test, 'r').readlines()
    clf, surrogate_corpus, surrogate_corpus_labels = None, None, None

    src_train_read = open(src_train, 'r').readlines()
    tgt_train_read = open(tgt_train, 'r').readlines()

    surrogate_corpus = src_train_read + tgt_train_read
    surrogate_corpus_labels = [0 for s in src_train_read] + [1 for s in tgt_train_read]
    surrogate_corpus = list(zip(surrogate_corpus, surrogate_corpus_labels))
    random.shuffle(surrogate_corpus)
    surrogate_corpus_labels = [l for (s,l) in surrogate_corpus]
    surrogate_corpus = [s for (s,l) in surrogate_corpus]
    
    surrogate_class = MLPSurrogate if clf_type=='mlp' else LogisticRegressionSurrogate
    surrogate_vectorizer = TfidfVectorizer if clf_vectorizer=='tf-idf' else CountVectorizer
    clf = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, clf_feat, clf_ngram_range, clf_max_feats), 1).fit(surrogate_corpus, surrogate_corpus_labels)
    


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

    