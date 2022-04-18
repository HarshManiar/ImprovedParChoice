from math import dist


def enum_idx(line, eligible_tags):
    line_idx = []
    for i in enumerate(line):
        if i[1][0] == '@' and i[1] in eligible_tags:
            line_idx.append(i[0])
    return line_idx

def distance(src_tokens, tgt_tokens, src_idx, tgt_idx):
    dist = 0
    if tgt_idx < src_idx:
        dist = src_idx-tgt_idx+1
    elif tgt_idx > src_idx:
        dist = tgt_idx-src_idx
    return dist

def transfer_context(src_tokens, tgt_tokens, eligible_tags):
    """
    Function to take in source and result lines and output the line that
    best aligns the two based on a selected loss function.

    Inputs:
        - src_tokens:         Line of interest in initial test dataset
        - tgt_tokens:         Line of interest in transformer/parchoice output
    Outputs:
        - ctxt:             Revised line with context transferred
    """
    at_src = enum_idx(src_tokens, eligible_tags)
    at_tgt = enum_idx(tgt_tokens, eligible_tags)
    ctxt = tgt_tokens
    if len(at_tgt) > 0 and len(at_src) > 0:
        distance_matrix = [[0 for _ in range(len(at_tgt))] for _ in range(len(at_src))]
        for i in range(len(at_src)):
            for j in range(len(at_tgt)):
                distance_matrix[i][j] = distance(src_tokens, tgt_tokens, at_src[i], at_tgt[j])

        # Find optimal token pairs in greedy manner
        pairings = []
        end_matrix = [[-1 for _ in range(len(at_tgt))] for _ in range(len(at_src))]
        max_dist = 0
        while distance_matrix != end_matrix:
            for i in range(len(distance_matrix)):
                for j in range(len(distance_matrix[0])):
                    if distance_matrix[i][j] == max_dist:
                        pairings.append((at_src[i], at_tgt[j]))
                        distance_matrix[i] = [-1 for i in range(len(distance_matrix[0]))]
                        for k in range(len(distance_matrix)):
                            distance_matrix[k][j] = -1
            max_dist += 1

        # Swap any token pairs
        for i, j in pairings:
            ctxt[j] = src_tokens[i]

        # Prepend all unused tokens after transformation
        if len(at_src) > len(at_tgt):
            s_lst = []
            for i in pairings:
                s_lst.append(i[0])
            for i in at_src:
                if i not in s_lst:
                    ctxt = [src_tokens[i]] + ctxt

        # Delete all tokens not used by original source
        if len(at_tgt) > len(at_src):
            new_ctxt = []
            for i in ctxt:
                if i[0] == '@' and i not in src_tokens:
                    pass
                else:
                    new_ctxt.append(i)
            ctxt = new_ctxt


    # When tags are in original but not in transformed copy
    elif len(at_tgt) == 0:

        # Prepend all tokens from src to tgt
        ctxt = tgt_tokens
        for i in at_src:
            ctxt = [src_tokens[i]] + ctxt

    # When tags are in new copy but not in original
    else:

        # Delete all tokens from tgt
        ctxt = tgt_tokens
        new_ctxt = []
        for i in ctxt:
            if i[0] == '@' and i not in src_tokens:
                pass
            else:
                new_ctxt.append(i)
        ctxt = new_ctxt

    return ctxt


def preserve_context(src_addr, tgt_addr, src_train_addr, tgt_train_addr, output='output.txt'):
    """
    Function that allows twitter @ tags to be carried over from the initial
    input to the result of the transformer to preserve more context.

    Inputs:
        - src_addr:         Initial test dataset
        - tgt_addr:         Result after transformer/parchoice
        - src_train_addr:   Source training set
        - tgt_train_addr:   Target training set
    Outputs:
        - tgt:              Revised target for context
    """

    with open(src_addr, 'r', encoding="utf8") as src, open(tgt_addr, 'r', encoding="utf8") as tgt, open(src_train_addr, 'r', encoding="utf8") as src_train, open(tgt_train_addr, 'r', encoding="utf8") as tgt_train:
        src = src.readlines()
        tgt = tgt.readlines()
        src_train = src_train.readlines()
        tgt_train = tgt_train.readlines()

        # Find all eligible tags for context preservation. Ignore others not in training set.
        eligible_tags = {}
        for line in src_train:
            tokens = line.split()
            for token in tokens:
                if token[0] == '@' and len(token) > 1:
                    eligible_tags[token] = 1
        for line in tgt_train:
            tokens = line.split()
            for token in tokens:
                if token[0] == '@' and len(token) > 1:
                    eligible_tags[token] = 1

        ctxt_list = []
        for src_line, tgt_line in zip(src, tgt):
            src_tokens = src_line.split()
            tgt_tokens = tgt_line.split()
            ctxt = transfer_context(src_tokens, tgt_tokens, eligible_tags)
            ctxt_list.append(' '.join(ctxt))

        with open(output, 'w', encoding="utf-8") as file:
            for line in ctxt_list:
                file.write(line + '\n')

        return ctxt_list

if __name__ == '__main__':
    print(preserve_context('sample_outputs/gold_elon_to_trump.txt', 'sample_outputs/rev_elon_to_trump.txt', 'twitter_data/train/elon_musk.txt', 'twitter_data/train/donald_trump.txt')[0])
    print(preserve_context('sample_outputs/gold_trump_to_elon.txt', 'sample_outputs/rev_trump_to_elon.txt', 'twitter_data/train/donald_trump.txt', 'twitter_data/train/elon_musk.txt')[0])