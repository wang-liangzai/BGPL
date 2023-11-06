import argparse
import os
import sys
import logging
import pickle
from functools import partial
import time
from tqdm import tqdm
from collections import Counter
import random
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor

from transformers import AdamW, T5Tokenizer
from t5 import MyT5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

from aos_data_utils import ABSADataset, task_data_list, cal_entropy
from aos_const import *
from aos_data_utils import read_line_examples_from_file, force_tokens, cate_list, force_words
from aos_eval_utils import compute_scores, extract_spans_para

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--data_path", default="../data/", type=str)
    parser.add_argument(
        "--task",
        default='aste',
        type=str,
        help="The name of the task")
    parser.add_argument(
        "--dataset",
        default='laptop14',
        type=str,
        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument(
        "--eval_data_split",
        default='test',
        choices=["test", "dev"],
        type=str,
    )
    parser.add_argument("--model_name_or_path",
                        default='../T5-base',
                        type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir",
                        default='outputs/aos/temp',
                        type=str,
                        help="Output directory")
    parser.add_argument("--load_ckpt_name",
                        default=None,
                        type=str,
                        help="load ckpt path")
    parser.add_argument("--do_train",
                        default=False,
                        help="Whether to run training.")
    parser.add_argument(
        "--do_inference",
        default=True,
        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=5,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--multi_path", action='store_true')
    parser.add_argument("--num_path", default=1, type=int)
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--save_top_k", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--single_view_type",
                    default="rank",
                    choices=["rank", "rand", "heuristic"],
                    type=str)
    parser.add_argument("--ctrl_token",
                        default="post",
                        choices=["post", "pre", "none"],
                        type=str)
    parser.add_argument("--sort_label",
                        action='store_true',
                        help="sort tuple by order of appearance")
    parser.add_argument("--load_path_cache",
                        action='store_true',
                        help="load decoded path from cache")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--multi_task", action='store_true')
    parser.add_argument("--constrained_decode",
                        default=True,
                        help='constrained decoding when evaluating')
    parser.add_argument('--agg_strategy', type=str, default='vote', choices=['vote', 'rand', 'heuristic', 'pre_rank', 'post_rank'])
    parser.add_argument("--data_ratio",
                        default=1,
                        type=float,
                        help="low resource data ratio")

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs/aos'):
        os.mkdir('./outputs/aos')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return args


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, config, tfm_model, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['tfm_model'])
        self.config = config
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(input_ids=batch["source_ids"],
                       attention_mask=batch["source_mask"],
                       labels=lm_labels,
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        # get f1
        outs = self.model.generate(input_ids=batch['source_ids'],
                                   attention_mask=batch['source_mask'],
                                   max_length=self.config.max_seq_length,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   num_beams=1)

        dec = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in outs.sequences
        ]
        target = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        final_calculation = []
        scores, _, _ = compute_scores(dec, target, final_calculation, verbose=False)
        f1 = torch.tensor(scores['f1'], dtype=torch.float64)

        # get loss
        loss = self._step(batch)

        if stage:
            self.log(f"{stage}_loss",
                     loss,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)
            self.log(f"{stage}_f1",
                     f1,
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          eps=self.config.adam_epsilon)
        scheduler = {
            "scheduler":
            get_linear_schedule_with_warmup(optimizer,
                                            **self.config.lr_scheduler_init),
            "interval":
            "step",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        print("load training data.")
        train_dataset = ABSADataset(tokenizer=self.tokenizer,
                                    task_name=args.task,
                                    data_name=args.dataset,
                                    data_type="train",
                                    top_k=self.config.top_k,
                                    args=self.config,
                                    max_len=self.config.max_seq_length)

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            drop_last=True
            if args.data_ratio > 0.3 else False, # don't drop on few-shot
            shuffle=True,
            num_workers=2)

        return dataloader

    def val_dataloader(self):
        val_dataset = ABSADataset(tokenizer=self.tokenizer,
                                  task_name=args.task,
                                  data_name=args.dataset,
                                  data_type="dev",
                                  top_k=self.config.num_path,
                                  args=self.config,
                                  max_len=self.config.max_seq_length)
        return DataLoader(val_dataset,
                          batch_size=self.config.eval_batch_size,
                          num_workers=2)

    @staticmethod
    def rindex(_list, _value):
        return len(_list) - _list[::-1].index(_value) - 1

    def prefix_allowed_tokens_fn(self, task, data_name, source_ids, batch_id,
                                 input_ids):
        """
        Constrained Decoding
        # ids = self.tokenizer("text", return_tensors='pt')['input_ids'].tolist()[0]
        """
        if not os.path.exists('./force_tokens.json'):
            dic = {"cate_tokens":{}, "all_tokens":{}, "sentiment_tokens":[], 'special_tokens':[]}
            for task in force_words.keys():
                dic["all_tokens"][task] = {}
                for dataset in force_words[task].keys():
                    cur_list = force_words[task][dataset]
                    tokenize_res = []
                    for w in cur_list:
                        tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0])
                    dic["all_tokens"][task][dataset] = tokenize_res
            for k,v in cate_list.items():
                tokenize_res = []
                for w in v:
                    tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
                dic["cate_tokens"][k] = tokenize_res
            sp_tokenize_res = []
            for sp in ['great', 'ok', 'bad']:
                sp_tokenize_res.extend(self.tokenizer(sp, return_tensors='pt')['input_ids'].tolist()[0])
            dic['sentiment_tokens'] = sp_tokenize_res
            special_tokens_tokenize_res = []
            for w in ['[O','[A','[S','[C','[SS']:
                special_tokens_tokenize_res.extend(self.tokenizer(w, return_tensors='pt')['input_ids'].tolist()[0]) 
            special_tokens_tokenize_res = [r for r in special_tokens_tokenize_res if r != 784]
            dic['special_tokens'] = special_tokens_tokenize_res
            import json
            with open("force_tokens.json", 'w') as f:
                json.dump(dic, f, indent=4)

        to_id = {
            'OT': [667],
            'AT': [188],
            'SP': [134],
            'AC': [254],
            'SS': [4256],
            'EP': [8569],
            '[': [784],
            ']': [908],
            'it': [34],
            'null': [206,195]
        }

        left_brace_index = (input_ids == to_id['['][0]).nonzero()
        right_brace_index = (input_ids == to_id[']'][0]).nonzero()
        num_left_brace = len(left_brace_index)
        num_right_brace = len(right_brace_index)
        last_right_brace_pos = right_brace_index[-1][
            0] if right_brace_index.nelement() > 0 else -1
        last_left_brace_pos = left_brace_index[-1][
            0] if left_brace_index.nelement() > 0 else -1
        cur_id = input_ids[-1]

        if cur_id in to_id['[']:
            return force_tokens['special_tokens']
        elif cur_id in to_id['AT'] + to_id['OT'] + to_id['EP'] + to_id['SP'] + to_id['AC']:  
            return to_id[']']  
        elif cur_id in to_id['SS']:  
            return to_id['EP'] 

        # get cur_term
        if last_left_brace_pos == -1:
            return to_id['['] + [1]   # start of sentence: [
        elif (last_left_brace_pos != -1 and last_right_brace_pos == -1) \
            or last_left_brace_pos > last_right_brace_pos:
            return to_id[']']  # ]
        else:
            cur_term = input_ids[last_left_brace_pos + 1]

        ret = []
        if cur_term in to_id['SP']:  # SP
            ret = force_tokens['sentiment_tokens'][task]
        elif cur_term in to_id['AT']:  # AT
            force_list = source_ids[batch_id].tolist()
            if task != 'aste':  
                force_list.extend(to_id['it'] + [1])  
            ret = force_list  
        elif cur_term in to_id['SS']:
            ret = [3] + to_id[']'] + [1]
        elif cur_term in to_id['AC']:  # AC
            ret = force_tokens['cate_tokens'][data_name]
        elif cur_term in to_id['OT']:  # OT
            force_list = source_ids[batch_id].tolist()
            if task == "acos":
                force_list.extend(to_id['null'])  # null
            ret = force_list
        else:
            raise ValueError(cur_term)

        if num_left_brace == num_right_brace:
            ret = set(ret)
            ret.discard(to_id[']'][0]) # remove ]
            for w in force_tokens['special_tokens']:
                ret.discard(w)
            ret = list(ret)
        elif num_left_brace > num_right_brace:
            ret += to_id[']'] 
        else:
            raise ValueError
        ret.extend(to_id['['] + [1]) # add [
        return ret

def reorder_group(group_data):
    return [group_data[0], group_data[2], group_data[1]]

def reorder_data(data_list):
    num_elements_per_group = 3
    num_groups = len(data_list) // num_elements_per_group
    reordered_data_list = [element for group in [reorder_group(data_list[i:i+num_elements_per_group]) for i in range(0, len(data_list), num_elements_per_group)] for element in group]

    return reordered_data_list
def evaluate(model, task, data, data_type):
    """
    Compute scores given the predictions and gold labels
    """
    tasks, datas, sents, _ = read_line_examples_from_file(
        f'../data/{task}/{data}/{data_type}.txt', task, data, lowercase=False)
    prob_file = './prob/aos_prob.txt'
    prob_f = os.path.join(args.output_dir, "prob_file.txt")
    terms_prob_file = os.path.join(args.output_dir, "terms_prob_file.txt")
    outputs, targets, probs, probs_list, final_results_list, final_triple_prob, terms_prob = [], [], [], [], [], [], []
    num_path = args.num_path
    if task in ['aste', 'tasd']:
        num_path = min(5, num_path)

    cache_file = os.path.join(
        args.output_dir, "result_{}{}{}_{}_path{}_beam{}.pickle".format(
            "best_" if args.load_ckpt_name else "",
            "cd_" if args.constrained_decode else "", task, data, num_path,
            args.beam_size))
    if args.load_path_cache:
        with open(cache_file, 'rb') as handle:
            (outputs, targets, probs) = pickle.load(handle)
    else:
        dataset = ABSADataset(model.tokenizer,
                              task_name=task,
                              data_name=data,
                              data_type=data_type,
                              top_k=num_path,
                              args=args,
                              max_len=args.max_seq_length)
        data_loader = DataLoader(dataset,
                                 batch_size=args.eval_batch_size,
                                 num_workers=2)
        device = torch.device('cuda:0')
        model.model.to(device)
        model.model.eval()

        error_list = []    
        for batch in tqdm(data_loader):
            probs = []
            # beam search
            outs = model.model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=args.max_seq_length,
                num_beams=args.beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=partial(
                    model.prefix_allowed_tokens_fn, task, data,
                    batch['source_ids']) if args.constrained_decode else None,
            )

            
            temp_list = []    
            for i in range(len(outs.sequences)):   
                value_list = ''    
                for j in range(len(outs.scores)):
                    if j == 0:
                        prob = outs.scores[0][i]    
                    elif j == 1:
                        temp_prob = outs.scores[j][i]    
                        prob = torch.stack([prob, temp_prob], dim=0)
                    else:
                        temp_prob = outs.scores[j][i]
                        prob = torch.cat((prob, temp_prob.unsqueeze(0)), dim=0)
                probs.append(prob)    
                for k in range(len(outs.scores)):
                    value = torch.max(F.softmax(probs[i][k], dim=-1))    
                    value_list = value_list + str(round(value.item(), 4)) + ' '
                value_list += '\n'
                temp_list.append(value_list)
            dec = [
                model.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in outs.sequences
            ]
            target = [
                model.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in batch["target_ids"]
            ]
            outputs.extend(dec)
            targets.extend(target)
            probs_list.extend(temp_list)   
            
            triple_num, triple_index, temp_index, final_results, triple_prob = [], [], [], [], []
            start_index = None
            last_index = None
            for i in range(len(target)):    
                results = []
                num = len(dec[i].split('[SSEP]'))    
                triple_num.append(num)
                sequence = outs.sequences[i]   
                for i, value in enumerate(sequence):
                    if value == 908:
                        if start_index is None:
                            start_index = i
                    elif value == 784 and start_index is not None:
                        end_index = i
                        content_indices = list(range(start_index, end_index - 1))
                        content = [sequence[idx] for idx in content_indices if sequence[idx] != '']
                        if content:
                            results.append(content_indices)
                        start_index = None
                    elif value == 1:
                        last_index = i

                # Check if there is content between the last 908 and 1
                if start_index is not None and last_index is not None and last_index > start_index:
                    content_indices = list(range(start_index, last_index - 1))
                    content = [sequence[idx] for idx in content_indices if sequence[idx] != '']
                    if content:
                        results.append(content_indices)

                final_results.append(results)
            final_results_list.extend(final_results)
            
            for i in range(len(target)):
                temp1 = temp_list[i].split()
                temp2 = final_results[i]
                new_temp = [[temp1[idx] for idx in sublist] for sublist in temp2]
                if len(new_temp) / 3 != triple_num[i]:
                  error_list.append(len(outputs) - args.eval_batch_size + i)
                triple_prob.append(new_temp)
            final_triple_prob.extend(triple_prob)
        
        for i in error_list:
            temp3 = outputs[i]
            split_parts = temp3.split('[SSEP]')

            
            filtered_parts = [part for part in split_parts if all(['[O]' in part, '[A]' in part, '[S]' in part])]
            for j, part in enumerate(split_parts):
                num = 0
                num_A = part.count('[A]')
                num_O = part.count('[O]')
                num_S = part.count('[S]')
                if num_A != 1 or num_O != 1 or num_S != 1:
                    num = num_A + num_O + num_S
                if num > 3:
                    error_index = j
                    del final_results_list[i][(error_index)*3:(error_index)*3+num]
                    del final_triple_prob[i][(error_index)*3:(error_index)*3+num]

                if '[O]' not in part or '[A]' not in part or '[S]' not in part or len(part.split('] [')) > 1:
                    if '[O]' in part and '[A]' in part:
                        error_index = j
                        del final_results_list[i][(error_index)*3:(error_index)*3+2]
                        del final_triple_prob[i][(error_index)*3:(error_index)*3+2]
                    elif '[O]' in part and '[S]' in part:
                        error_index = j
                        del final_results_list[i][(error_index)*3:(error_index)*3+2]
                        del final_triple_prob[i][(error_index)*3:(error_index)*3+2]
                    elif '[A]' in part and '[S]' in part:
                        error_index = j
                        del final_results_list[i][(error_index)*3:(error_index)*3+2]
                        del final_triple_prob[i][(error_index)*3:(error_index)*3+2]
                    else:
                        error_index = j
                        del final_results_list[i][(error_index)*3:(error_index)*3+1]
                        del final_triple_prob[i][(error_index)*3:(error_index)*3+1]
                if '[O]' in part and '[A]' in part and '[S]' in part:
                    str1 = part.split('[O] [A]')
                    str2 = part.split('[A] [S]')
                    str3 = part.split('[S]')
                    if len(str1) > 1 or len(str2) > 1 or str3[1] == '':
                            error_index = j
                            del final_results_list[i][(error_index)*3:(error_index)*3+3]
                            del final_triple_prob[i][(error_index)*3:(error_index)*3+3]

            new_str = ''.join(filtered_parts)
            new_str = new_str.replace(' [O]', '[SSEP] [O]')

            outputs[i] = new_str
        for i in range(len(final_triple_prob)):
            split_num = len(outputs[i].split('[SSEP]'))  
            prob_num = len(str(final_triple_prob[i]).split('],'))
            if final_triple_prob[i] != '' and prob_num == 3*split_num:
                terms_prob.append(reorder_data(final_triple_prob[i]))
            else:
                terms_prob.append('')
                final_triple_prob[i] = ''

        _calculation, final_calculation = [], []
        for i in range(len(final_triple_prob)):
            temp_calculation = []
            for element in final_triple_prob[i]:
                calculation = ''
                if len(element) > 1:
                    multiplied_value = 1.0
                    for value in element:
                        multiplied_value *= float(value)
                    root = math.pow(multiplied_value, 1 / len(element))
                    calculation = ''.join(str([round(root, 4)]))
                else:
                    calculation = ''.join(str([round(float(element[0]), 4)]))
                temp_calculation.append(calculation)
            _calculation.append(temp_calculation)
        for i in range(len(_calculation)):
            if _calculation[i] != '': 
                final_calculation.append(reorder_data(_calculation[i]))
            else:
                final_calculation.append('')

        with open(prob_f, "w") as file1:
            for item in final_calculation:
                file1.write(str(item) + '\n')
        file1.close()
        with open(terms_prob_file, "w") as file2:
            for item in terms_prob:
                file2.write(str(item) + '\n')

        with open(prob_file, 'w') as file:
            for i in range(len(probs_list)):
                file.write(str(probs_list[i]))
                file.write('\n')
        # save outputs and targets
        with open(cache_file, 'wb') as handle:
            pickle.dump((outputs, targets, probs), handle)

    if args.multi_path:
        targets = targets[::num_path]

        # get outputs
        _outputs = outputs # backup
        outputs = [] # new outputs
        if args.agg_strategy == 'post_rank':
            inputs = [ele for ele in sents for _ in range(num_path)]
            assert len(_outputs) == len(inputs), (len(_outputs), len(inputs))
            preds = [[o] for o in _outputs] 
            model_path = os.path.join(args.output_dir, "final")
            scores = cal_entropy(inputs, preds, model_path, model.tokenizer)

        for i in range(0, len(targets)):
            o_idx = i * num_path
            multi_outputs = _outputs[o_idx:o_idx + num_path]

            if args.agg_strategy == 'post_rank':
                multi_probs = scores[o_idx:o_idx + args.num_path]
                assert len(multi_outputs) == len(multi_probs)

                sorted_outputs = [i for _,i in sorted(zip(multi_probs,multi_outputs))]
                outputs.append(sorted_outputs[0])
                continue
            elif args.agg_strategy == "pre_rank":
                outputs.append(multi_outputs[0])
                continue
            elif args.agg_strategy == 'rand':
                outputs.append(random.choice(multi_outputs))
                continue
            elif args.agg_strategy == 'heuristic':
                # aspect term > opinion term = aspect category > sentiment polarity
                optim_orders_all = get_orders_all()
                heuristic_orders = get_orders_heuristic()
                index = optim_orders_all[task][data].index(heuristic_orders[task][0])
                outputs.append(multi_outputs[index])
                # at, ot/ac, sp
                continue
            elif args.agg_strategy == 'vote':
                all_quads = []
                for s in multi_outputs:
                    all_quads.extend(
                        extract_spans_para(seq=s, seq_type='pred'))

                output_quads = []
                counter = dict(Counter(all_quads))
                for quad, count in counter.items():
                    # keep freq >= num_path / 2
                    if count >= len(multi_outputs) / 2:
                        output_quads.append(quad)

                # recover output
                output = []
                for q in output_quads:
                    ac, at, sp, ot = q
                    if tasks[i] == "aste":
                        if 'null' not in [at, ot, sp]:  # aste has no 'null', for zero-shot only
                            output.append(f'[A] {at} [O] {ot} [S] {sp}')

                    elif tasks[i] == "tasd":
                        output.append(f"[A] {at} [S] {sp} [C] {ac}")

                    elif tasks[i] in ["asqp", "acos"]:
                        output.append(f"[A] {at} [O] {ot} [S] {sp} [C] {ac}")

                    else:
                        raise NotImplementedError

                target_quads = extract_spans_para(seq=targets[i],
                                                seq_type='gold')

                if sorted(target_quads) != sorted(output_quads):
                    print("task, data:", tasks[i], datas[i])
                    print("target:", sorted(target_quads))
                    print('output:', sorted(output))
                    print("sent:", sents[i])
                    print("counter:", counter)
                    print("output quads:", output)
                    print("multi_path:", multi_outputs)
                    print()

                # if no output, use the first path
                output_str = " [SSEP] ".join(
                    output) if output else multi_outputs[0]

                outputs.append(output_str)

    # stats
    labels_counts = Counter([len(l.split('[SSEP]')) for l in outputs])
    print("pred labels count", labels_counts)

    scores, all_labels, all_preds = compute_scores(outputs,
                                                   targets,
                                                   final_calculation,
                                                   verbose=True)
    return scores, all_labels, all_preds


def train_function(args):

    # training process
    scores = {}
    if args.do_train:
        print("\n", "=" * 30, f"NEW EXP: {args.task} on {args.dataset}",
              "=" * 30, "\n")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)

        # sanity check
        # show one sample to check the code and the expected output
        print(f"Here is an example (from the dev set):")
        dataset = ABSADataset(tokenizer=tokenizer,
                              task_name=args.task,
                              data_name=args.dataset,
                              data_type='train',
                              top_k=args.top_k,
                              args=args,
                              max_len=args.max_seq_length)
        for i in range(0, min(1, len(dataset))):
            data_sample = dataset[i]
            print(
                'Input :',
                tokenizer.decode(data_sample['source_ids'],
                                 skip_special_tokens=True))
            print('Input :',
                  tokenizer.convert_ids_to_tokens(data_sample['source_ids']))
            print(
                'Output:',
                tokenizer.decode(data_sample['target_ids'],
                                 skip_special_tokens=True))
            print()

        print("\n****** Conduct Training ******")

        # initialize the T5 model
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, local_files_only=True if args.model_name_or_path != "t5-base" else False)
        model = T5FineTuner(args, tfm_model, tokenizer)

        # load data
        train_loader = model.train_dataloader()

        # config optimizer
        t_total = ((len(train_loader.dataset) //
                    (args.train_batch_size * max(1, args.n_gpu))) //
                   args.gradient_accumulation_steps *
                   float(args.num_train_epochs))

        args.lr_scheduler_init = {
            "num_warmup_steps": args.warmup_steps,
            "num_training_steps": t_total
        }

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename='{epoch}-{val_f1:.2f}-{val_loss:.2f}',
            monitor='val_f1',
            mode='max',
            save_top_k=args.save_top_k,
            save_last=False)

        early_stop_callback = EarlyStopping(monitor="val_f1",
                                            min_delta=0.00,
                                            patience=100,
                                            verbose=True,
                                            mode="max")
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # prepare for trainer
        train_params = dict(
            accelerator="gpu",
            devices=1,
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[
                checkpoint_callback, early_stop_callback,
                TQDMProgressBar(refresh_rate=10), lr_monitor
            ],
        )

        trainer = pl.Trainer(**train_params)

        trainer.fit(model)

        # save the final model
        model.model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print("Finish training and saving the model!")

    if args.do_inference:

        print("\n****** Conduct inference on trained checkpoint ******")

        # initialize the T5 model from previous checkpoint
        print(f"Load trained model from {args.output_dir}")
        print(
            'Note that a pretrained model is required and `do_true` should be False'
        )
        model_path = os.path.join(args.output_dir, "final")
        # model_path = args.model_name_or_path  # for loading ckpt

        tokenizer = T5Tokenizer.from_pretrained(model_path)
        tfm_model = MyT5ForConditionalGeneration.from_pretrained(model_path)
        model = T5FineTuner(args, tfm_model, tokenizer)

        if args.load_ckpt_name:
            ckpt_path = os.path.join(args.output_dir, args.load_ckpt_name)
            print("Loading ckpt:", ckpt_path)
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint["state_dict"])

        log_file_path = os.path.join(args.output_dir, "result.txt")

        # compute the performance scores
        with open(log_file_path, "a+") as f:
            config_str = f"seed: {args.seed}, beam: {args.beam_size}, constrained: {args.constrained_decode}\n"
            print(config_str)
            f.write(config_str)

            if args.multi_task:
                f1s = []
                for task in task_data_list:
                    for data in task_data_list[task]:
                        scores, all_labels, all_preds = evaluate(model, task, data, data_type=args.eval_data_split)
                        print(task, data, scores)
                        exp_results = "{} {} precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                            args.eval_data_split, args.agg_strategy, scores['precision'], scores['recall'],
                            scores['f1'])
                        f.write(f"{task}: \t{data}: \t{exp_results}\n")
                        f.flush()
                        f1s.append(scores['f1'])
                f.write(f"Average F1: \t{sum(f1s) / len(f1s)}\n")
                f.flush()
            else:
                scores, all_labels, all_preds = evaluate(model,
                                  args.task,
                                  args.dataset,
                                data_type=args.eval_data_split)
                result_txt = './outputs/aos/result_aos.txt'

                with open(result_txt, 'w') as file:
                    for i in range(len(all_preds)):
                        file.write(str(all_preds[i]))
                        file.write('\n')
                file.close()

                exp_results = "{} {} precision: {:.2f} recall: {:.2f} F1 = {:.2f}".format(
                    args.eval_data_split, args.agg_strategy, scores['precision'], scores['recall'], scores['f1'])
                print(exp_results)
                f.write(exp_results + "\n")
                f.flush()


if __name__ == '__main__':
    print("start training aos")
    args = init_args()
    set_seed(args.seed)
    train_function(args)

    