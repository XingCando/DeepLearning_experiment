import os
import torch
import numpy as np
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
import time


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


def format_params(num):
    """将参数数量格式化为 K/M/B 单位，方便阅读"""
    for unit in ['', 'K', 'M', 'B']:
        if num < 1024:
            return f"{num:.2f}{unit}"
        num /= 1024
    return f"{num:.2f}T"

    
# 自定义平均损失计算器，替代废弃的 torchnet.AverageValueMeter
class AverageValueMeter:
    def __init__(self):
        self.reset()  # 初始化时重置所有参数

    def reset(self):
        # 重置：总和、数量、均值、方差
        self.sum = 0
        self.count = 0
        self.avg = 0

    def add(self, value):
        # 传入当前损失，累加计算
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count  # 实时计算平均值

    def value(self):
        # 返回平均损失（兼容原接口格式）
        return [self.avg]
    
    
def get_eval_score(references, hypotheses):
    # 定义评估指标（固定7个指标）
    # hypotheses的形状为[num_samples, len_caption],
    # 形如[[idx1,idx2,idx3,...],[idx1,idx2,idx3,...],...]
    # references的形状为[num_samples, 5, len_caption], 元素也是词索引
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    # 所有样本的生成翻译, [num_samples, len_caption]
    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    # 所有样本的参考翻译, [num_samples, 5]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    all_scores = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        # BLEU：scores_i 是4个列表 → 用extend追加4组分数
        if isinstance(scores_i[0], list):
            all_scores.extend(scores_i)
        # 其他指标：scores_i 是1组分数列表 → 用append追加整个列表（不拆分）
        else:
            all_scores.append(scores_i)
        #print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict, all_scores