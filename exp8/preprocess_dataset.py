import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type = str, default = 'RSICD', help= 'the name of the dataset')
parser.add_argument('--input_captions_json', type = str, default = './RSICD/dataset_rsicd.json', help = 'input captions json file')
parser.add_argument('--save_dir', type = str, default = './RSICD/')
parser.add_argument('--word_count_threshold', default=3, type=int)

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}

def main(args):
    input_captions_json = args.input_captions_json
    input_vocab_json = ''
    output_vocab_json = 'vocab.json'
    output_vocab_frequency = 'vocab_freq.json'
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir + 'tokens/')):
        os.makedirs(os.path.join(save_dir + 'tokens/'))
    print('Loading captions')

    if args.dataset == 'RSICD':
        with open(input_captions_json, 'r') as f:
            data = json.load(f)
        # Read image paths and captions for each image
        max_length = -1
        all_cap_tokens = []
        for img in data['images']:
            captions = []    
            for c in img['sentences']:
                # Update word frequency
                assert len(c['raw']) > 0, 'error: some image has no caption'
                captions.append(c['raw'])
            tokens_list = []
            for cap in captions:
                cap_tokens = tokenize(cap,
                                    add_start_token=True,
                                    add_end_token=True,
                                    punct_to_keep=[';', ','],
                                    punct_to_remove=['?', '.'])
                if len(cap_tokens) == 2:
                    print(f"图像{img['filename']}没有参考描述，删除!")
                tokens_list.append(cap_tokens)
                max_length = max(max_length, len(cap_tokens))
            all_cap_tokens.append((img['filename'], img['split'], tokens_list)) #（图像文件名，训练/验证/测试，五句描述, 每句描述都是词元列表）

        # Then save the tokenized captions in txt
        print('Saving captions')
        train_w_mode = 'w'
        val_w_mode = 'w'
        test_w_mode = 'w'
        for img, split, tokens_list in all_cap_tokens:
            i = img.split('.')[0]
            token_len = len(tokens_list)
            tokens_list = json.dumps(tokens_list)
            f = open(os.path.join(save_dir + 'tokens/' + i + '.txt'), 'w') #./data/LEVIR_CC/tokens/train_000001.txt
            f.write(tokens_list)
            f.close()


        #Considering each image pair has 5 annotations, two strategies can be adopted to generate list for training:
        # a: creating training list with a self-defined token_id[0:4], each token list corresponds to specific captions;
        # or b: randomly select one of the five captions during training;
            '''
            if i.split('_')[0] == 'train':
               f = open(os.path.join(save_dir + 'train' + '.txt'), train_w_mode)
               f.write(img + '\n')
               f.close
               train_w_mode = 'a'
            '''
                
            # 对同一张图片复制五张，末尾编号0，1，2，3，4
            if split == 'train':
                f = open(os.path.join(save_dir + 'train' + '.txt'), train_w_mode)
                for j in range(token_len):
                    f.write(img + '-' + str(j) + '\n')
                f.close
                train_w_mode = 'a'

            elif split == 'val':
                f = open(os.path.join(save_dir + 'val' + '.txt'), val_w_mode)
                f.write(img + '\n')
                f.close()
                val_w_mode = 'a'

            elif split == 'test':
                f = open(os.path.join(save_dir + 'test' + '.txt'), test_w_mode)
                f.write(img + '\n')
                f.close()
                test_w_mode = 'a'


    print('sentence max_length of the dataset:', max_length)
    # Either create the vocab or load it from disk
    if input_vocab_json == '':
        print('Building vocab')
        vocab_idx, word_freq = build_vocab(all_cap_tokens, args.word_count_threshold)
    else:
        print('Loading vocab')
        with open(input_vocab_json, 'r') as f:
            vocab_idx = json.load(f)
    if output_vocab_json != '':
        with open(os.path.join(save_dir + output_vocab_json), 'w') as f:
            json.dump(vocab_idx, f)
    if output_vocab_frequency != '':
        with open(os.path.join(save_dir + output_vocab_frequency), 'w') as f:
            json.dump(word_freq, f)


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True, punct_to_keep=None, punct_to_remove=None):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, f'{delim}{p}')

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    # 过滤空 token
    tokens = [t for t in s.split(delim) if t != '']

    # 防止空列表访问错误
    if len(tokens) == 0:
        if add_start_token and add_end_token:
            return ['<START>', '<END>']
        elif add_start_token:
            return ['<START>']
        elif add_end_token:
            return ['<END>']
        else:
            return []

    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')

    return tokens

def build_vocab(sequences, min_token_count=1):#Calculate the number of independent words and tokenize vocab
    token_to_count = {}
    # 对所有描述中的词进行计数
    for it in sequences:
        for seq in it[2]:
            for token in seq:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    # 特殊字符索引
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    # 为每一个词建立字符索引
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count > min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx, token_to_count

def encode(seq_tokens, token_to_idx, allow_unk=False):
    # 将描述文本编码为词索引序列
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
