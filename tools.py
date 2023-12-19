import re
import os
import pdb
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from ngcse.models import NGBert, NGRoberta

import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained('/home/LAB/limx/download/model/bert-base-uncased')

# >>>>>>>>>> generate data
def split_corpus(data_path):
    with open(data_path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]
    file_prefix = os.path.splitext(data_path)[0]

    indices = list(range(len(lines)))
    random.shuffle(indices)
    indices_heldout, indices_train = indices[:len(indices) // 10], indices[len(indices) // 10:]
    indices_heldout.sort()
    indices_train.sort()
    lines_heldout = [lines[idx] for idx in indices_heldout]
    lines_train = [lines[idx] for idx in indices_train]

    with open(f"{file_prefix}_heldout.jsonl", 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in lines_heldout))
    with open(f"{file_prefix}_train.jsonl", 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in lines_train)) 

def token_shuffle(corpus_path: str, dump_path: str, tokenizer):
    with open(corpus_path, 'r', encoding='utf8') as fi:
        rows = [json.loads(row) for row in fi]
    for row in tqdm(rows):
        tokens = tokenizer.encode(row['sentence'], add_special_tokens=False)
        random.shuffle(tokens)
        row['5'] = tokenizer.decode(tokens)
    with open(dump_path, 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))

def token_cutoff(corpus_path: str, dump_path: str, tokenizer, ratio: float = 0.05):
    with open(corpus_path, 'r', encoding='utf8') as fi:
        rows = [json.loads(row) for row in fi]
    for row in tqdm(rows):
        tokens = tokenizer.encode(row['sentence'], add_special_tokens=False)
        indices = list(range(len(tokens)))
        random.shuffle(indices)
        indices = indices[round(len(indices) * ratio): ]
        indices.sort()
        n_tokens = [tokens[idx] for idx in indices]
        row['5'] = tokenizer.decode(n_tokens)
    with open(dump_path, 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
# <<<<<<<<<< generate data

# >>>>>>>>>> collect statistcs
def collect_loss_and_eval(log_path: str):
    with open(log_path, 'r', encoding='utf8') as fi:
        content = fi.read()
    losses = re.findall(r"'loss': ([\d\.]+),?", content)
    stsbs = re.findall(r"'eval_stsb_spearman': ([\d\.]+),", content)
    sickrs = re.findall(r"'eval_sickr_spearman': ([\d\.]+),", content)
    avgs = re.findall(r"'eval_avg_sts': ([\d\.]+),", content)
    data = {
        'step': [125 * (i + 1) for i in range(len(losses))], # FIXME: check this, now start with 125
        'loss': [loss for loss in losses],
        'stsb': [stsb for stsb in stsbs],
        'sickr': [sickr for sickr in sickrs],
        'avg': [avg for avg in avgs]
    }

    df = pd.DataFrame(data)
    path_names = list(os.path.split(log_path))
    path_names[-1] = f"{path_names[-1].split('.')[0]}.csv"
    df.to_csv(os.path.join(*path_names), index=False)

def align_data(base_dir, domains, patterns, pattern2split, num, dump_dir):
    file_num = {domain: 0 for domain in domains}
    data = {domain: {} for domain in domains}
    for domain in domains:
        for pattern in patterns:
            data_path = os.path.join(base_dir, domain, pattern, 'heldout.jsonl')
            if not os.path.exists(data_path):
                data_path = os.path.join(base_dir, domain, pattern, 'heldout_bert.jsonl')
                if not os.path.exists(data_path):
                    data_path = None

            if data_path is not None:
                file_num[domain] += 1
                with open(data_path, 'r', encoding='utf8') as fi:
                    lines = [json.loads(line) for line in fi]
                for line in lines:
                    if line['split'] == pattern2split(pattern):
                        if line['sentence'] not in data[domain]:
                            data[domain][line['sentence']] = []
                        data[domain][line['sentence']].append((pattern, line))

    coexist_data = {domain:[
        pnls for pnls in data[domain].values() 
        if len(pnls) == file_num[domain]
    ][:num] for domain in domains}
    
    results = {domain: {} for domain in domains}
    for domain in domains:
        results[domain] = {pattern: [] for pattern, _ in coexist_data[domain][0]}
        for pnls in coexist_data[domain]:
            for pattern, line in pnls:
                results[domain][pattern].append(line)
    
    for domain in results:
        for pattern in results[domain]:
            dump_path = os.path.join(dump_dir, domain, pattern)
            os.makedirs(os.path.join(dump_dir, domain, pattern), exist_ok=True)
            with open(os.path.join(dump_path, f'aligned_{num}.jsonl'), 'w', encoding='utf8') as fo:
                fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in results[domain][pattern]))
    
def load_stsb(split: str, mode: str = 'positive', max_num: int = 1000):
    stsb = load_dataset(r'/home/LAB/limx/download/corpus/stsbenchmark-sts', split=split)

    if mode == 'positive':
        return [(row['sentence1'], row['sentence2']) for row in stsb if row['score'] >= 4.][:max_num]
    elif mode == 'negative':
        return [(row['sentence1'], row['sentence2']) for row in stsb if row['score'] <= 1.][:max_num]
    elif mode == 'single':
        return [row['sentence1'] for row in stsb][:max_num]
    elif mode == 'all':
        res = []
        for row in stsb:
            if row['score'] >= 4. or row['score'] <= 1.:
                res.append(row['sentence1'])
                res.append(row['sentence2'])
        return res[:max_num]
    else:
        raise NotImplementedError

def load_jsonl(data_path: str, mode: str = 'positive', 
        positive_key: str = None, mid_key: str = None, negative_key: str = None, max_num: int = 1000):
    with open(data_path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]
    
    if mode == 'positive':
        if positive_key is None:
            positive_key = 'sentence'
        return [(line['sentence'], line[positive_key]) for line in lines][:max_num]
    elif mode == 'negative':
        if negative_key is None:
            other_lines = lines.copy()
            random.shuffle(other_lines)
            return [(line['sentence'], other_line['sentence']) for line, other_line in zip(lines, other_lines)][:max_num]
        else:
            return [(line['sentence'], line[negative_key]) for line in lines][:max_num]
    elif mode == 'single':
        res = [line['sentence'] for line in lines][:max_num]
        return res
    elif mode == 'all':
        res = []
        for line in lines:
            res.append(line['sentence'])
            if positive_key is not None:
                res.append(line[positive_key])
            if mid_key is not None:
                res.append(line[mid_key])
            if negative_key is not None:
                res.append(line[negative_key])
        return res[:max_num]
    else:
        raise NotImplementedError


def load_model(model_path: str, is_bert: bool = True):
    
    # Load transformers' model checkpoint
    if is_bert:
        model = NGBert.from_pretrained(model_path)
    else:
        model.NGRoberta.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    return model, tokenizer, device

def batcher(sentences, model, tokenizer, device, max_length=512):
    # Tokenization
    if max_length is not None:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=True
        )
    else:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )

    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    return pooler_output.cpu()

def cal_embeddings(model, tokenizer, device, sentences, batch_size=1024):
    paired = isinstance(sentences[0], tuple)

    b_sentences = []
    features = []
    # for sentence in tqdm(sentences, desc="Calculate Embeddings"):
    for sentence in sentences:
        if paired:
            for l_sentence in sentence:
                b_sentences.append(l_sentence)
        else:
            b_sentences.append(sentence)

        if len(b_sentences) >= batch_size:
            features.append(batcher(b_sentences, model, tokenizer, device))
            b_sentences = []
    if len(b_sentences) > 0:
        features.append(batcher(b_sentences, model, tokenizer, device))
        b_sentences = []
    
    if paired:
        for i, b_features in enumerate(features):
            features[i] = b_features.reshape(-1, len(sentences[0]), b_features.shape[-1])
    
    features = torch.cat(features, dim=0)
    return features

def collect_embeddings(ckpt_dir, sentences,
    base_model: str = '/home/LAB/limx/download/model/bert-base-uncased'):
    save_name = 'ckpts_features_paired' if isinstance(sentences[0], tuple) else 'ckpts_features'
    save_name = f'{save_name}_{len(sentences)}.pt'

    if save_name in os.listdir(ckpt_dir):
        print('Cached ckpt_features found, loading.')
        ckpts_features = torch.load(os.path.join(ckpt_dir, save_name))
    else:
        ckpts = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'checkpoint' in ckpt]
        
        ckpts_features = {
            0: cal_embeddings(*load_model(base_model), sentences)
        }
        for ckpt in tqdm(ckpts, desc="Collect Embeddings"):
            step = int(os.path.split(ckpt)[-1][len('checkpoint-'): ])
            model, tokenizer, decive = load_model(ckpt)
            ckpts_features[step] = cal_embeddings(model, tokenizer, decive, sentences)
        
        torch.save(ckpts_features, os.path.join(ckpt_dir, save_name))

    return ckpts_features

def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def cal_alignment_and_uniformity(ckpt_dir, load_fn, first_arg, align_kwargs, uniform_kwargs,
    cal_aligment=True, cal_uniformity=True):

    # alignment
    if cal_aligment:
        print('Calculate Alignment:\n')
        sentences = load_fn(first_arg, **align_kwargs)
        ckpts_features = collect_embeddings(ckpt_dir, sentences)
        ckpts_alignment = {}
        for step, features in tqdm(ckpts_features.items(), desc="Calculate Alignment"):
            ckpts_alignment[step] = align_loss(features[:, 0], features[:, 1])

        alignments = sorted(((step, alignment) for step, alignment in ckpts_alignment.items()), key=lambda x: x[0])
        with open(os.path.join(ckpt_dir, 'alignments'), 'w', encoding='utf8') as fo:
            fo.write('\n'.join(f'{step}\t\t{alignment}' for step, alignment in alignments))
        print(f'Average Alignment: {sum(alignment for _, alignment in alignments) / len(alignments)}')
        # for step, alignment in alignments:
        #     print(f'{step}\t\t{alignment}')

    # uniformity
    if cal_uniformity:
        print('Calculate Uniformity:\n')
        sentences = load_fn(first_arg, **uniform_kwargs)
        ckpts_features = collect_embeddings(ckpt_dir, sentences)
        ckpts_uniformity = {}
        for step, features in tqdm(ckpts_features.items(), desc="Calculate Uniformity"):
            ckpts_uniformity[step] = uniform_loss(features)
        
        uniformities = sorted(((step, uniformity) for step, uniformity in ckpts_uniformity.items()), key=lambda x: x[0])
        with open(os.path.join(ckpt_dir, 'uniformities'), 'w', encoding='utf8') as fo:
            fo.write('\n'.join(f'{step}\t\t{uniformity}' for step, uniformity in uniformities))
        print(f'Average Uniformity: {sum(uniformity for _, uniformity in uniformities) / len(uniformities)}')
        # for step, uniformity in uniformities:
        #     print(f'{step}\t\t{uniformity}')

def assemble_alignment_and_uniformity(ckpt_dir):

    with open(os.path.join(ckpt_dir, 'alignments'), 'r', encoding='utf8') as fi:
        alignments = sorted(((int(step), float(alignment)) for step, alignment in (row.strip().split() for row in fi)), key=lambda x: x[0])

    with open(os.path.join(ckpt_dir, 'uniformities'), 'r', encoding='utf8') as fi:
        uniformities = sorted(((int(step), float(uniformity)) for step, uniformity in (row.strip().split() for row in fi)), key=lambda x: x[0])
    
    anu = {'step':[], 'alignment': [], 'uniformity': []}
    for sa, su in zip(alignments, uniformities):
         assert sa[0] == su[0]
         anu['step'].append(sa[0])
         anu['alignment'].append(sa[1])
         anu['uniformity'].append(su[1])
    df = pd.DataFrame(anu)
    df.to_csv(os.path.join(ckpt_dir, 'alignment_and_uniformity.csv'), index=False)
# <<<<<<<<<< collect statistcs

if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    '''# >>>>>>>>>> collect loss
    domains = ['nli', 'wiki']
    patterns = ['dropout', 'token_cutoff', 'token_shuffle', 'nli', 'nli_whn', 'sts', 'sts_whn', 'sts_ht', 'sup']
    base_dir = r'results/data_domain/bert'
    for domain in domains:
        for pattern in patterns:
            log_path = os.path.join(base_dir, domain, pattern)
            if os.path.exists(log_path):
                collect_loss_and_eval(log_path)
    '''# <<<<<<<<<< collect loss



    '''# >>>>>>>>>> align_data
    align_data(
        r'data/data_domain', ['nli', 'wiki'], 
        ['dropout', 'token_cutoff', 'token_shuffle', 'nli', 'sts', 'sup'],
        lambda pattern: 'aigen' if pattern == 'sts' or pattern =='nli' else 'other',
        1000, r'data/data_domain/align'
    )
    '''# <<<<<<<<<< align_data

    '''# >>>>>>>>>> calculate alignment and uniformity
    base_dir = r'results/runs/data_domain/bert'
    domains = ['nli', 'wiki']
    patterns = ['dropout', 'token_cutoff', 'token_shuffle', 'nli_whn', 'sts_whn', 'sts_ht', 'sup'] # 'nli', 'sts', 
    triplets = []
    for domain in domains:
        for pattern in patterns:
            ckpt_dir = os.path.join(base_dir, f"{domain}.{pattern}")
            if os.path.exists(ckpt_dir):
                triplets.append((domain, pattern, ckpt_dir))

    pattern2kwargs = {
        'stsb': {'align_kwargs': {'mode': 'positive'}, 'uniform_kwargs': {'mode': 'all'}}, # {'mode': 'all'}
        'dropout': {'align_kwargs': {'mode': 'positive'}, 'uniform_kwargs': {'mode': 'single'}},
        'token_cutoff': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                         'uniform_kwargs': {'mode': 'single'}},
        'token_shuffle': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                          'uniform_kwargs': {'mode': 'single'}},
        'nli': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'negative_key': '4'}},
        'nli_whn': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                    'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'negative_key': '4'}},
        'sts': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'negative_key': '3'}}, 
        'sts_whn': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                    'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'negative_key': '3'}},
        'sts_ht': {'align_kwargs': {'mode': 'positive', 'positive_key': '4'}, 
                   'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'mid_key': '4', 'negative_key': '3'}},
        'sup': {'align_kwargs': {'mode': 'positive', 'positive_key': '5'}, 
                'uniform_kwargs': {'mode': 'all', 'positive_key': '5', 'negative_key': '4'}},
    }
    pattern2data_type = {pattern: pattern for pattern in patterns}
    pattern2data_type.update({'nli_whn': 'nli', 'sts_whn': 'sts', 'sts_ht': 'sts'})

    data_base_dir = r'data/data_domain/align'
    for domain, pattern, ckpt_dir in triplets:
        print(f'**{domain.upper()}.{pattern.upper()}**')
        for label, load_fn, arg, kwargs in zip(
            ['train', 'eval'], [load_jsonl, load_stsb],
            [os.path.join(data_base_dir, domain, pattern2data_type[pattern], 'aligned_1000.jsonl'), 'validation'],
            [pattern2kwargs[pattern], pattern2kwargs['stsb']]
        ):
            print(f'*{label.upper()}*')
            dst_dir = os.path.join(ckpt_dir, label)
            if os.path.exists(dst_dir):
                print('This split seems to be done before, please check yourself.')
                continue
                
            cal_alignment_and_uniformity(ckpt_dir, load_fn, arg, **kwargs)
            assemble_alignment_and_uniformity(ckpt_dir)

            os.makedirs(dst_dir)
            file_names = ['alignment_and_uniformity.csv', 'alignments', 'uniformities']
            file_names.extend(file_name for file_name in os.listdir(ckpt_dir) \
                if 'ckpts_features_' in file_name or 'ckpts_features_paired_' in file_name)
            for file_name in file_names:
                os.system(f'mv {os.path.join(ckpt_dir, file_name)} {dst_dir}')
    '''# <<<<<<<<<< alignment and uniformity