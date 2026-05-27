# 【必须放在所有 import 代码的最前面！】
import sys
from pathlib import Path
import numpy as np
import os
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import argparse
import json
from tqdm import tqdm
from RSICD.RSICD import RSICDDataset
#from data.WHU_CC.WHUCC import WHUCDCDataset
from model_encoder import CNNTransformerModel
from model_decoder import TransformerDecoder
from utils import *
from PIL import Image
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_captions(args, word_map, bleu_4s, img_names, hypotheses, references):
    """
    将模型生成的文本（hypotheses）和参考文本（references）保存为 JSON 文件。
    
    :param args: 包含配置信息的对象，例如模型名称、数据集划分等。
    :param word_map: 单词到索引的映射字典。
    :param hypotheses: 模型生成的文本，是一个嵌套列表，形状为 [num_samples, sequence_length]。
    :param references: 参考文本，是一个嵌套列表，形状为 [num_samples, num_references, sequence_length]。
    """

    # 初始化两个字典，用于存储图像文件名，生成描述和参考描述
    json_file = {} 
    kkk = -1
    # 遍历生成文本（hypotheses）
    for bleu4, img_name, hypo, ref in zip(bleu_4s, img_names, hypotheses, references):
        kkk += 1  # 计数器加 1，表示当前处理的样本索引
        #source_line = ""  # 初始化空字符串，用于存储当前源文本的句子
        hypo_line = "" # 初始化空字符串，用于存储当前生成描述
        json_file[str(kkk)] = {}
        # 遍历当前生成描述中的每个单词索引
        for word_idx in hypo:
            word = get_key(word_map, word_idx)
            hypo_line += word[0] + " "
        # 将当前生成描述的句子存储到json_file对应样本索引的字典中
        json_file[str(kkk)]["图像filename"] = img_name  
        json_file[str(kkk)]["生成描述"] = hypo_line

        # 初始化当前样本的参考文本列表
        json_file[str(kkk)]['参考描述'] = []
        # 遍历当前样本的每个参考文本
        for sentence in ref:
            line_repo = ""  # 初始化空字符串，用于存储当前参考文本的句子
            # 遍历当前参考文本序列中的每个单词索引
            for word_idx in sentence:
                # 通过 word_map 字典将单词索引映射回单词
                word = get_key(word_map, word_idx)
                # 将单词拼接到 line_repo 字符串中，并添加一个空格
                line_repo += word[0] + " "
            # 将当前参考文本的句子存储到 reference_json_file 字典中
            json_file[str(kkk)]['参考描述'].append(line_repo)

        json_file[str(kkk)]["BLEU-4"] = 100*bleu4
    if os.path.exists(args.result_path)==False:
            os.makedirs(args.result_path)
    
    # 将生成文本保存为 JSON 文件
    with open(args.result_path + '_' + args.network + '_sorted_captions.json', 'w') as f:
        json.dump(json_file, f)  # 将 result_json_file 字典写入文件
        

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value] #遍历字典，查找与 value 匹配的键


def main(args):
    """
    Testing.
    """
    visualized_log = open(os.path.join(args.log_dir, 'test.log'), 'w')
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Load checkpoint
    snapshot_full_path = args.checkpoint#os.path.join(args.savepath, args.checkpoint)
    checkpoint = torch.load(snapshot_full_path)
    args.result_path = os.path.join(args.result_path, os.path.basename(snapshot_full_path).replace('.pth', ''))

    encoder = CNNTransformerModel(network=args.backbone, 
                                  emb_dim=args.embed_dim, 
                                  num_heads=args.encoder_num_heads, 
                                  num_layers=args.encoder_num_layers, 
                                  dropout=args.dropout, 
                                  h_feat=args.h_feat, 
                                  w_feat=args.w_feat)
    decoder = TransformerDecoder(embed_dim=args.embed_dim, 
                                 vocab_size=len(word_vocab), 
                                 max_lengths=args.max_length, 
                                 word_vocab=word_vocab, 
                                 n_head=args.decoder_num_heads, 
                                 n_layers=args.decoder_num_layers, 
                                 dropout=args.dropout)


    encoder.load_state_dict(checkpoint['encoder_dict'])
    #encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    # Move to GPU, if available
    encoder.eval()
    encoder = encoder.to(device)
    '''
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    '''
    decoder.eval()
    decoder = decoder.to(device)
    print('load model success!')

    test_dataset = RSICDDataset(args.backbone, args.data_folder, args.list_path, 'test', token_folder = args.token_folder, word_vocab = word_vocab, max_length = args.max_length, allow_unk = args.allow_unk)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Epochs
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    img_names = list()

    with torch.no_grad():
        # Batches
        for ind, batch_data in enumerate(
                tqdm(test_dataloader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
            # Move to GPU, if available
            img = batch_data['img']
            token_all = batch_data['token_all']
            token_all_len = batch_data['token_all_len']
            img_name = batch_data['name']
            img = img.to(device)
            token_all = token_all.squeeze(0).to(device)
            # Forward prop.
            feat = encoder(img)
            # for captioning
            seq = decoder.sample_beam(feat, k=3)
            except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<PAD>']}
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
                        img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in except_tokens]
            hypotheses.append(pred_seq)
            img_names.append(img_name[0])
            assert len(references) == len(hypotheses)

            # save_captions(pred_caption, ref_captions, hypotheses[-1], references[-1], name, args.result_path)
        test_time = time.time() - test_start_time

        # Fast test during the training
        # Calculate evaluation scores
        cc_score_dict, all_scores = get_eval_score(references, hypotheses)
        Bleu_1 = 100*cc_score_dict['Bleu_1']
        Bleu_2 = 100*cc_score_dict['Bleu_2']
        Bleu_3 = 100*cc_score_dict['Bleu_3']
        Bleu_4 = 100*cc_score_dict['Bleu_4']
        Meteor = 100*cc_score_dict['METEOR']
        Rouge = 100*cc_score_dict['ROUGE_L']
        Cider = 100*cc_score_dict['CIDEr']
        print_log('Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                  'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), visualized_log)
        
        # 提取所有样本的 Bleu_4 分数
        bleu_4_scores = all_scores[3]
        # 使用 zip 将四个列表打包成一个包含元组的列表
        # 格式: [(score1, image_name1, hyp1, ref1), (score2, image_name2, hyp2, ref2), ...]
        combined_data = list(zip(bleu_4_scores, img_names, hypotheses, references))
        # 按 Bleu_4 分数（即元组的第一个元素 x[0]）从高到低进行降序排序
        sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)
        # 使用 zip(*...) 将排序好的数据解包回四个独立的元组，并转回列表
        sorted_bleu_4, sorted_img_name, sorted_hypotheses, sorted_references = map(list, zip(*sorted_data))

        save_captions(args, word_vocab, sorted_bleu_4, sorted_img_name, sorted_hypotheses, sorted_references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Captioning')

    # Data parameters
    parser.add_argument('--sys', default='linux', choices=('linux'), help='system')
    parser.add_argument('--data_folder', type=str, default='./RSICD/images', help='图像数据集路径')
    parser.add_argument('--list_path', type=str, default='./RSICD/', help='数据集根目录')
    parser.add_argument('--token_folder', type=str, default='./RSICD/tokens/', help='数据集tokens文件路径')
    parser.add_argument('--vocab_file', type=str, default='vocab', help='词典文件名')
    parser.add_argument('--max_length', type=int, default=36, help='the max_length of sentences')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    # Test
    parser.add_argument('--checkpoint', default='./checkpoint/cnn-transformer_best_epo_1_Bleu4_459304.pth', help='path to checkpoint, None if none.')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')
    parser.add_argument('--workers', type=int, default=0,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    # Validation
    parser.add_argument('--result_path', default="./results/")
    
    # model parameters
    parser.add_argument('--network', default='cnn-transformer', help=' define the backbone encoder to extract features')
    parser.add_argument('--backbone', type=str, default='resnet101', help='网络骨架名称')
    parser.add_argument('--embed_dim', type=int, default=512, help='模型的全局隐藏维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--encoder_num_heads', type=int, default=8, help='编码器Transformer注意力头数')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='解码器Transformer注意力头数')
    parser.add_argument('--encoder_num_layers', type=int, default=3, help='编码器Transformer层数')
    parser.add_argument('--decoder_num_layers', type=int, default=2, help='解码器Transformer层数')
    parser.add_argument('--h_feat', type=int, default=7, help='图像特征图高度')
    parser.add_argument('--w_feat', type=int, default=7, help='图像特征图宽度')

    # 运行和保存设置
    parser.add_argument('--log_dir', default='logs', type=str, help='Directory for TensorBoard logs')
    args = parser.parse_args()
    print('list_path:', args.list_path)

    main(args)
