# 【必须放在所有 import 代码的最前面！】
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import argparse
import time
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from model_encoder import CNNTransformerModel
from model_decoder import TransformerDecoder
from RSICD.RSICD import RSICDDataset
from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_noam_schedule_with_warmup(optimizer, d_model, warmup_steps, factor=1.0, last_epoch=-1):
    """
    创建 Noam (Inverse Square Root) 学习率调度器。
    
    参数:
        optimizer: PyTorch 优化器 (建议使用 Adam/AdamW)
        d_model: Transformer 模型的隐藏层维度 (例如 512)
        warmup_steps: 预热步数 (例如 4000)
        factor: 学习率缩放因子，用于整体调大或调小学习率曲线
        last_epoch: 恢复训练时的 epoch/step 索引，默认为 -1 (从头训练)
    """
    def lr_lambda(step):
        # 避免 step=0 时出现除零错误或计算出 0，将 step 限制最小为 1
        step = max(1, step) 
        
        # Noam 核心公式: d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        scale = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        
        return factor * scale

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(args, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler, train_dataloader, criterion, writer, loss_meter, visualized_log, total_steps):
    loss_meter.reset()
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for idx, batch_data in enumerate(train_dataloader):
        # Move to GPU, if available
        img = batch_data['img']
        token = batch_data['token']
        token_len = batch_data['token_len']
        img = img.to(device)
        token = token.to(device)
        token_len = token_len.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # Forward prop.
        feat = encoder(img) #(batch, 49, 768)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(feat, token, token_len)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # Calculate loss
        loss = criterion(scores, targets.to(torch.int64))
        
        loss.backward()
        # 可调参数 1 可以改为 其他值进行尝试，梯度裁剪
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_scheduler.step()
        decoder_scheduler.step()

        loss_meter.add(loss.item())
        total_steps += 1
        if total_steps % args.log_step == 0:
            current_lr = encoder_optimizer.param_groups[0]['lr']
            print_log(
                'Train Epoch: {}/{}, {}/{} ({:.0f}%), Step: {}, Loss: {:.3f}, Current LR: {:.4f}'.format(
                    epoch,
                    args.epochs,
                    idx,
                    len(train_dataloader),
                    idx * 100 / len(train_dataloader),
                    total_steps,
                    loss.item(),
                    current_lr
                ), visualized_log
            )
            writer.add_scalar('LR/train', current_lr, total_steps)

    #avg_loss = train_loss / len(dataloader)
    print_log(f'Epoch:{epoch} | train_Loss: {loss_meter.avg:.3f}', visualized_log)
    writer.add_scalar('Loss/train', loss_meter.avg, epoch)
    return total_steps

            
def valid_loss_epoch(epoch, encoder, decoder, val_dataloader, criterion, writer, visualized_log):
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(val_dataloader)):
            # Move to GPU, if available
            img = batch_data['img']
            token = batch_data['token']
            token_len = batch_data['token_len']
            img = img.to(device)
            token = token.to(device)
            token_len = token_len.to(device)
            # Forward prop.
            feat = encoder(img) #(batch, 49, 768)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(feat, token, token_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
    
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            # Calculate loss
            loss = criterion(scores, targets.to(torch.int64))

            val_loss += loss.item()
            
    print_log(f'Epoch:{epoch} | val_Loss: {val_loss/len(val_dataloader):.3f}', visualized_log)
    writer.add_scalar('Loss/val', val_loss/len(val_dataloader), epoch)
    
    return val_loss / len(val_dataloader)


def valid_bleu_epoch(epoch, encoder, decoder, val_dataloader, word_vocab, writer, visualized_log):
    encoder.eval()
    decoder.eval()
    references = list()  # references (true captions) for calculating BL,EU-4 score
    hypotheses = list()  # hypotheses (predictions)
    with torch.no_grad():
        for ind, batch_data in enumerate(
                    tqdm(val_dataloader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
            # if ind == 20:
            #     break
            # Move to GPU, if available
            # (imgA, imgB, token_all, token_all_len, _, _, _)
            img = batch_data['img']
            token_all = batch_data['token_all']
            token_all_len = batch_data['token_all_len']
            img = img.to(device)
            token_all = token_all.squeeze(0).to(device)
            # Forward prop.
            if encoder is not None:
                feat = encoder(img)
            seq = decoder.sample(feat, k=1)

            # for captioning
            except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<PAD>']}
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],img_token))  # remove <start> and pads
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in except_tokens]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
        # 计算模型生成序列的各项指标分数，用于评估模型训练的进程
        cc_score_dict, _ = get_eval_score(references, hypotheses)
        Bleu_1 = 100*cc_score_dict['Bleu_1']
        Bleu_2 = 100*cc_score_dict['Bleu_2']
        Bleu_3 = 100*cc_score_dict['Bleu_3']
        Bleu_4 = 100*cc_score_dict['Bleu_4']
        Meteor = 100*cc_score_dict['METEOR']
        Rouge = 100*cc_score_dict['ROUGE_L']
        Cider = 100*cc_score_dict['CIDEr']
        print_log('Epoch:{0: } | ''Caption_Validation:\n' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                    'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                    .format(epoch, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), visualized_log)
        writer.add_scalar('BLEU-4/val', Bleu_4, epoch)
        
    return Bleu_4


# 主执行函数
def main(args):
    best_bleu4 = 0
    last_no_improved = 0
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.net}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=log_path)
    visualized_log = open(os.path.join(args.log_dir, 'train.log'), 'w')
    # --- 数据加载与预处理 ---
    print_log('==> Preparing and loading data..', visualized_log)
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    train_dataset = RSICDDataset(args.backbone, args.data_folder, args.list_path, 'train', token_folder = args.token_folder, word_vocab = word_vocab, max_length = args.max_length, allow_unk = args.allow_unk)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    #trg_word2idx, trg_idx2word = train_dataloader.trg_word2id, train_dataloader.trg_id2word
    
    val_dataset = RSICDDataset(args.backbone, args.data_folder, args.list_path, 'val', token_folder = args.token_folder, word_vocab = word_vocab, max_length = args.max_length, allow_unk = args.allow_unk)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # 实例化模型并将其移动到指定设备
    print_log('==> Building model..', visualized_log)
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
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # 优化器设计
    encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=args.encoder_lr,weight_decay=args.weight_decay)
    decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=args.decoder_lr,weight_decay=args.weight_decay)
    '''
    # 初始化Noam调度器 (factor=1.0 是原版论文的默认缩放)
    Total_step = args.epochs * len(train_dataloader)
    encoder_lr_scheduler = get_noam_schedule_with_warmup(
        optimizer=encoder_optimizer, 
        d_model=args.embed_dim, 
        warmup_steps=min(4000, Total_step//10),
        factor=args.scheduler_factor
    )
    decoder_lr_scheduler = get_noam_schedule_with_warmup(
        optimizer=decoder_optimizer, 
        d_model=args.embed_dim, 
        warmup_steps=min(4000, Total_step//10),
        factor=args.scheduler_factor
    )
    '''


    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=5,
                                                                          gamma=1.0)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5,
                                                                    gamma=1.0)

    encoder_cnn_learnable_params = sum(p.numel() for p in encoder.cnn_encoder.parameters() if p.requires_grad)
    # 正确写法（通过参数名称筛选，安全无报错）
    encoder_trans_learnable_params = sum(
        p.numel() 
        for name, p in encoder.named_parameters() 
        if p.requires_grad and not name.startswith("cnn_encoder")
    )
    decoder_learnable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    # 输出结果
    print_log(f"CNN特征提取器可学习参数总数: {encoder_cnn_learnable_params} ({format_params(encoder_cnn_learnable_params)})", visualized_log)
    print_log(f"Transformer编码器层可学习参数总数: {encoder_trans_learnable_params} ({format_params(encoder_trans_learnable_params)})", visualized_log)
    print_log(f"Transformer解码器层可学习参数总数: {decoder_learnable_params} ({format_params(decoder_learnable_params)})", visualized_log)
    
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageValueMeter()
    total_steps = 0
    # 训练和验证循环
    for epoch in range(args.epochs):
        if last_no_improved == 10:
            print_log(f"模型在验证集上的BLEU-4指标已经连续10个epoch没有提升，训练停止!", visualized_log)
            break
        total_steps = train_epoch(args, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, encoder_lr_scheduler, decoder_lr_scheduler, train_dataloader, criterion, writer, loss_meter, visualized_log, total_steps)
        
        val_loss = valid_loss_epoch(epoch, encoder, decoder, val_dataloader, criterion, writer, visualized_log)
        bleu_4 = valid_bleu_epoch(epoch, encoder, decoder, val_dataloader, word_vocab, writer, visualized_log)
        
        print_log(f"\nEpoch {epoch} 完成. Val Loss: {val_loss:.3f}, Val BlEU-4: {bleu_4:.2f}", visualized_log)
        
        if bleu_4 > best_bleu4:
            print_log('Saving best model..', visualized_log)
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            state = {'encoder_dict': encoder.state_dict(),
                     'decoder_dict': decoder.state_dict()
                     }
            best_bleu4 = bleu_4
            metric = f'Bleu4_{round(10000 * best_bleu4)}'
            torch.save(state, os.path.join(args.checkpoint_dir, f'{args.net}_best_epo_{epoch}_{metric}.pth'))
            #scheduler.step()
            last_no_improved = 0
            print_log(f"Current best BLEU-4: {best_bleu4:.2f}", visualized_log)
        else:
            last_no_improved += 1
            
    writer.close()
    final_model_path = "cnn-transformer_final.pth"
    # 如果训练提前停止，我们仍然保存最后的模型状态，但最佳模型已经在过程中保存了
    state = {'encoder_dict': encoder.state_dict(),
             'decoder_dict': decoder.state_dict()
            }
    torch.save(state, os.path.join(args.checkpoint_dir,final_model_path))
    print_log(f"训练完成, 最终模型保存至 {os.path.join(args.checkpoint_dir,final_model_path)}", visualized_log)
    
    
if __name__ == '__main__':
    # ----------------------------------
    # 1. 参数解析器 (args)
    # ----------------------------------
    parser = argparse.ArgumentParser(description='PyTorch tang shi Training')
    
    # 数据集参数
    parser.add_argument('--data_folder', type=str, default='./RSICD/images', help='图像数据集路径')
    parser.add_argument('--list_path', type=str, default='./RSICD/', help='数据集根目录')
    parser.add_argument('--token_folder', type=str, default='./RSICD/tokens/', help='数据集tokens文件路径')
    parser.add_argument('--vocab_file', type=str, default='vocab', help='词典文件名')
    parser.add_argument('--max_length', type=int, default=36, help='the max_length of sentences')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    # 训练超参数
    parser.add_argument('--encoder_lr', default=2e-4, type=float, help='encoder learning rate')
    parser.add_argument('--decoder_lr', default=2e-4, type=float, help='decoder learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--train_batchsize', default=64, type=int, help='Training batch size')
    parser.add_argument('--val_batchsize', default=1, type=int, help='Val batch size')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler_factor', type=float, default=1.0, help='学习率缩放因子，用于整体调大或调小学习率曲线')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    # 模型超参数
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
    parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, help='Directory for saving checkpoints')
    parser.add_argument('--net', default='cnn-transformer', type=str, help='Network name for saving checkpoints')
    parser.add_argument('--log_step', type=int, default=100, help='每隔 log_step 次打印一次训练记录')
    args = parser.parse_args()
    
    main(args)