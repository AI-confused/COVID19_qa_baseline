import json
from pathlib import Path
import time
import torch
import numpy as np
import random
import pandas as pd
import re
import itertools
from pandas.io.json._json import JsonReader
from typing import Callable, Dict, List, Generator, Tuple
from multiprocessing import Pool
import os
import operator
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset, Dataset
from collections import defaultdict
from rouge import Rouge
from elasticsearch import Elasticsearch


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class ElasticObj:
    def __init__(self, index_name,index_type,ip ="127.0.0.1"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        # 无用户名密码状态
        self.es = Elasticsearch([ip])

    def Get_Data_By_Body(self, question, k):
        doc = {
            "size": k, 
            "query": {
                "match": {
                  "passage": question
                }
              }
        }
        try:
            _searched = self.es.search(index=self.index_name, doc_type=self.index_type, body=doc)
            answers = []
            for item in _searched['hits']['hits']:
                answers.append((item['_source']['passage'], item['_source']['docid'])) 
            return answers

        except:
            print('search not exist')
            print(question)
            
            
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, docid, text_a, text_b, start=None, end=None):
        self.guid = guid
        self.docid = docid
        self.text_a = text_a
        self.text_b = text_b
        self.start = start
        self.end = end


class Feature(object):
    def __init__(self, example_id, doc_start, question_len, tokenized_to_original_index, input_tokens, input_ids, input_mask, segment_ids, bert_start_position=None, bert_end_position=None, label=None, passage=None,origin_start=None,origin_end=None,token_start=None,token_end=None,doc_end=None,doc_id=None,bert_doc_end=None, bert_answer_span=None, cls_logit=None, best_scores=None, result=None, new_score=None,doc_length=None,search_pred=None):
        self.example_id = example_id
        self.doc_id = doc_id
        self.passage = passage
        self.doc_start = doc_start
        self.doc_end = doc_end
        self.doc_length=doc_length
        self.bert_doc_end=bert_doc_end
        self.question_len = question_len
        self.tokenized_to_original_index = tokenized_to_original_index
        self.input_tokens=input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.bert_start_position = bert_start_position
        self.bert_end_position = bert_end_position
        self.bert_answer_span = bert_answer_span
        self.label = label        
        self.origin_start=origin_start
        self.origin_end=origin_end
        self.token_start=token_start
        self.token_end=token_end

        
def read_examples(input_file, doc_file, is_training, k=1, es_index="passages", es_ip="localhost"):
    df=pd.read_csv(input_file)
    doc = pd.read_csv(doc_file)
    if is_training==1: # train
        examples=[]
        for val in df[['id','docid','question','start_index','end_index']].values:
            passage = doc[doc['docid']==val[1]].values[0][-1]
            examples.append(InputExample(guid=val[0], docid=val[1], text_a=passage, text_b=val[2], start=val[3], end=val[-1]-1)) # [start, end]闭区间
    elif is_training==2: # eval
        examples=[]
        for val in df[['id','docid','question','start_index','end_index']].values:
            passage = doc[doc['docid']==val[1]].values[0][-1]
            examples.append(InputExample(guid=val[0], docid=val[1], text_a=passage, text_b=val[2], start=val[3], end=val[-1]-1)) # [start, end]闭区间
    else: # test
        obj = ElasticObj(es_index,"_doc",ip =es_ip)
        examples=[]
        for val in df[['id','question']].values:
            answers = obj.Get_Data_By_Body(val[1], k)
            for passage, docid in answers:
                examples.append(InputExample(guid=val[0], docid=docid, text_a=passage, text_b=val[1]))
    return examples



def convert_examples_to_features(example, tokenizer, max_seq_length, max_question_length, is_training):
  # tokenize question
    question_tokens = tokenizer.tokenize(example.text_b)[:max_question_length]
  # tokenize passage & build original to token index
    passage_words = list(example.text_a) # original passage text
    original_to_tokenized_index = []
    tokenized_to_original_index = []  
    passage_tokens = []  # tokenized passage text
    for i, word in enumerate(passage_words):
        original_to_tokenized_index.append(len(passage_tokens))
        sub_tokens = tokenizer.tokenize(word)
        for sub_token in sub_tokens:
            tokenized_to_original_index.append(i)
            passage_tokens.append(sub_token) 
    assert len(tokenized_to_original_index)==len(passage_tokens)
  # find answer span in passage_tokens [start_position, end_position]闭区间
    if is_training in [1,2]:
        start_position = original_to_tokenized_index[example.start]
        end_position = original_to_tokenized_index[example.end]

  # split passage_tokens
    max_doc_length = max_seq_length - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
    features = []
    selected_sentenses = []
    start_point = 0
    for i,token in enumerate(passage_tokens):#把文档按照句号拆分为片段
        if token == '。' or i == len(passage_tokens)-1:
            selected_sentenses.append({'tokens':passage_tokens[start_point:i+1], 'start':start_point,'end':i})#闭区间[start, end]
            start_point = i+1
    assert len(selected_sentenses)>0, f'len(selected_sentenses) is 0!'
    doc_tokens = []
    features_append_flag = 0
    i = 0
    multi_sentense_start, multi_sentense_end = 0,0
    feature_count = 0
    while i < len(selected_sentenses):   
        if len(selected_sentenses[i]['tokens'])+len(doc_tokens) <= max_doc_length:
            if not doc_tokens:
                multi_sentense_start = selected_sentenses[i]['start']
            doc_tokens += selected_sentenses[i]['tokens']

            if i == len(selected_sentenses)-1:
                feature_count += 1
                features_append_flag = 1
                multi_sentense_end = selected_sentenses[i]['end']#闭区间
                assert len(doc_tokens)<=max_doc_length, f'len(doc_tokens) too much:{len(doc_tokens)}'
                assert multi_sentense_end > multi_sentense_start
                if is_training in [1,2]:
                    if multi_sentense_start >= end_position or multi_sentense_end <= start_position:
                        start = -1
                        end = -1
                        label = 'unknown'
                    elif multi_sentense_start <= start_position and end_position <= multi_sentense_end:
                        #checked
                        start = start_position - multi_sentense_start + len(question_tokens) + 2
                        end = end_position - multi_sentense_start + len(question_tokens) + 2
                        label = 'known'
                    else:
                        #checked
                        start = max(start_position - multi_sentense_start + len(question_tokens) + 2, len(question_tokens) + 2)
                        end = min(end_position - multi_sentense_start + len(question_tokens) + 2, multi_sentense_end - multi_sentense_start + len(question_tokens) + 2)
                        label = 'known'
                    assert -1 <= start < max_seq_length-2, f'start position is out of range: {start}'
                    assert -1 <= end < max_seq_length-1, f'end position is out of range: {end}'
            i += 1

        elif len(selected_sentenses[i]['tokens'])+len(doc_tokens) > max_doc_length and doc_tokens:
            feature_count += 1
            multi_sentense_end = selected_sentenses[i]['start']-1
            assert 0 < len(doc_tokens)<=max_doc_length, f'len(doc_tokens) too much:{len(doc_tokens)}'
            assert multi_sentense_end > multi_sentense_start
            features_append_flag = 1
            #checked
            if is_training in [1,2]:
                if multi_sentense_start >= end_position or multi_sentense_end <= start_position:
                    start = -1
                    end = -1
                    label = 'unknown'
                elif multi_sentense_start <= start_position and end_position <= multi_sentense_end:
                    start = start_position - multi_sentense_start + len(question_tokens) + 2
                    end = end_position - multi_sentense_start + len(question_tokens) + 2
                    label = 'known'
                else:
                    start = max(start_position - multi_sentense_start + len(question_tokens) + 2, len(question_tokens) + 2)
                    end = min(end_position - multi_sentense_start + len(question_tokens) + 2, multi_sentense_end - multi_sentense_start + len(question_tokens) + 2)
                    label = 'known'
                assert -1 <= start < max_seq_length-2, f'start position is out of range: {start}'
                assert -1 <= end < max_seq_length-1, f'end position is out of range: {end}'

        elif len(selected_sentenses[i]['tokens']) > max_doc_length:
            for sentense_start in range(0, len(selected_sentenses[i]['tokens']), max_doc_length-100):
                sentense_end = min(sentense_start + max_doc_length-1, len(selected_sentenses[i]['tokens'])-1) #overlap
                multi_sentense_start = selected_sentenses[i]['start']+sentense_start
                multi_sentense_end = selected_sentenses[i]['start']+sentense_end
                #checked
                if is_training in [1,2]:
                    if multi_sentense_start >= end_position or multi_sentense_end <= start_position:
                        start = -1
                        end = -1
                        label = 'unknown'
                    elif multi_sentense_start <= start_position and end_position <= multi_sentense_end:
                        start = start_position - multi_sentense_start + len(question_tokens) + 2
                        end = end_position - multi_sentense_start + len(question_tokens) + 2
                        label = 'known'
                    else:
                        start = max(start_position - multi_sentense_start + len(question_tokens) + 2, len(question_tokens) + 2)
                        end = min(end_position - multi_sentense_start + len(question_tokens) + 2, sentense_end - sentense_start + len(question_tokens) + 2)
                        label = 'known'
                    assert -1 <= start < max_seq_length-2, f'start position is out of range: {start}'
                    assert -1 <= end < max_seq_length-1, f'end position is out of range: {end}'
                feature_count += 1
                doc_tokens = selected_sentenses[i]['tokens'][sentense_start:sentense_end+1]
                assert len(doc_tokens)<=max_doc_length, f'len(doc_tokens) too much:{len(doc_tokens)}'
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
                # 在bert的输入序列中，有效文档片段的结尾索引，闭区间
                bert_doc_end = len(input_tokens) - 1
                assert len(input_tokens)<=max_seq_length
                segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(doc_tokens) + 1)
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                input_mask = [1] * len(input_ids)
                #padding
                padding_length = max_seq_length - len(input_ids)
                input_ids += ([0] * padding_length)
                input_mask += ([0] * padding_length)
                segment_ids += ([0] * padding_length)
                if is_training==1: # train dataset
                    features.append(
                        Feature(
                            example_id=example.guid,
                            doc_id=example.docid,
                            doc_start=multi_sentense_start,
                            doc_end=multi_sentense_end,
                            doc_length=multi_sentense_end-multi_sentense_start,
                            question_len=len(question_tokens),
                            tokenized_to_original_index=tokenized_to_original_index,
                            input_tokens=input_tokens,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            bert_start_position=start, # bert input seq start position
                            bert_end_position=end,     # bert input seq end position
                            bert_answer_span=end-start,
                            label=label,
                            token_start=start_position,
                            token_end=end_position
                    ))
                elif is_training==2: # eval dataset
                    features.append(
                        Feature(
                            example_id=example.guid,
                            passage=example.text_a,
                            doc_start=multi_sentense_start, 
                            bert_doc_end=bert_doc_end,
                            question_len=len(question_tokens),
                            tokenized_to_original_index=tokenized_to_original_index,
                            input_tokens=input_tokens,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            bert_start_position=start, 
                            bert_end_position=end,
                            bert_answer_span=end-start,
                            label=label,
                            origin_start=example.start,
                            origin_end=example.end
                    ))
                else:# test
                    features.append(
                        Feature(
                            example_id=example.guid,
                            doc_id=example.docid,
                            passage=example.text_a,
                            doc_start=multi_sentense_start, 
                            bert_doc_end=bert_doc_end,
                            question_len=len(question_tokens),
                            tokenized_to_original_index=tokenized_to_original_index,
                            input_tokens=input_tokens,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                    ))
                doc_tokens = []
            i += 1

        if features_append_flag:
            features_append_flag = 0
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
            bert_doc_end = len(input_tokens) - 1
            assert len(input_tokens)<=max_seq_length
            segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(doc_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_mask = [1] * len(input_ids)
            #padding
            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            if is_training==1: # train dataset
                features.append(
                    Feature(
                        example_id=example.guid,
                        doc_id=example.docid,
                        doc_start=multi_sentense_start,
                        doc_end=multi_sentense_end,
                        doc_length=multi_sentense_end-multi_sentense_start,
                        question_len=len(question_tokens),
                        tokenized_to_original_index=tokenized_to_original_index,
                        input_tokens=input_tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        bert_start_position=start, # bert input seq start position
                        bert_end_position=end,     # bert input seq end position
                        bert_answer_span=end-start,
                        label=label,
                        token_start=start_position,
                        token_end=end_position
                ))
            elif is_training==2: # eval dataset
                features.append(
                    Feature(
                        example_id=example.guid,
                        passage=example.text_a,
                        doc_start=multi_sentense_start, 
                        bert_doc_end=bert_doc_end,
                        question_len=len(question_tokens),
                        tokenized_to_original_index=tokenized_to_original_index,
                        input_tokens=input_tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        bert_start_position=start, 
                        bert_end_position=end,
                        bert_answer_span=end-start,
                        label=label,
                        origin_start=example.start,
                        origin_end=example.end
                ))
            else:# test
                features.append(
                    Feature(
                        example_id=example.guid,
                        doc_id=example.docid,
                        passage=example.text_a,
                        doc_start=multi_sentense_start, 
                        bert_doc_end=bert_doc_end,
                        question_len=len(question_tokens),
                        tokenized_to_original_index=tokenized_to_original_index,
                        input_tokens=input_tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                ))
            doc_tokens = []

    assert len(features) == feature_count, f'example features not right!'

    return features

def pre_process(train_features: List[List[Feature]]):
    selected_features = []
    single_answer, multi_answer, has_unknown_sample = 0,0,0
    for index in range(len(train_features)): # num_examples
        annotated_known = list(filter(lambda example: example.label == 'known', train_features[index]))
        assert len(annotated_known)>=1, f'example dont has answer span!'
        annotated_unknown = list(filter(lambda example: example.label == 'unknown', train_features[index]))
            
        annotated = []
        if len(annotated_known)==1:
            single_answer += 1
            if annotated_unknown:
                has_unknown_sample += 1
                annotated = (annotated_known + [random.choice(annotated_unknown)])#随机选择一个负样本
            else:
                annotated = annotated_known
        else:
            multi_answer += 1
            annotated = annotated_known
            if annotated_unknown:
                has_unknown_sample += 1
                annotated += [random.choice(annotated_unknown)]
        assert len(annotated)>=1

        selected_features += annotated
    logger.info('*** answers analyze ***')
    logger.info('*** single answer num: {}'.format(single_answer))
    logger.info('*** multi answer num: {}'.format(multi_answer))
    logger.info('*** has_unknown_sample num: {}'.format(has_unknown_sample))
    return selected_features

    
class TextDataset(Dataset):
    def __init__(self, examples: List[Feature]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
      
    def __getitem__(self, index):
        return self.examples[index]

                      
def collate_fn(examples: List[Feature]):
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_mask = torch.stack([torch.tensor(example.input_mask, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)
    # output labels
    all_labels = ['known', 'unknown']
    start_positions = np.array([example.bert_start_position for example in examples])
    end_positions = np.array([example.bert_end_position for example in examples])
    class_labels = [all_labels.index(example.label) for example in examples]
    labels = [torch.LongTensor(start_positions),
              torch.LongTensor(end_positions),
              torch.LongTensor(class_labels)]

    return [input_ids, input_mask, segment_ids, labels]


def eval_collate_fn(examples: List[Feature]):
    input_ids = torch.stack([torch.tensor(example.input_ids, dtype=torch.long) for example in examples],0)
    input_mask = torch.stack([torch.tensor(example.input_mask, dtype=torch.long) for example in examples],0)
    segment_ids = torch.stack([torch.tensor(example.segment_ids, dtype=torch.long) for example in examples],0)

    return input_ids, input_mask, segment_ids, examples



class Result(object):
    """Stores results of all test data.
    """
    def __init__(self, is_testing=0):
        self.examples = {}
        self.results = {}
        self.best_scores = {}
        self.unknown_examples = {}
        self.class_labels = ['known', 'unknown']
        self.is_testing = is_testing
        
    @staticmethod
    def is_valid_index(example: Feature, index: List[int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if end_index - start_index < 1:#bert seq 中start&end相对位置正确
            return False
        #bert seq中start不在question范围，也不在padding的范围
        if start_index < example.question_len + 2 or start_index >= example.bert_doc_end-1:
            return False
        #bert seq中end不在question范围，也不在padding的范围
        if end_index <= example.question_len + 2 or end_index > example.bert_doc_end-1:
            return False
        return True
        
    def update(
        self,
        examples,
        start_preds,
        end_preds,
        class_preds,
        k
    ):
        #从分数top k的span中选择合适的span
        start_logits, start_index = torch.topk(start_preds, k, dim=1, largest=True, sorted=True)
        end_logits, end_index = torch.topk(end_preds, k, dim=1, largest=True, sorted=True)
        cls_logits = start_preds[:, 0] + end_preds[:, 0]
        indices = [[(int(start), int(end)) for start in start_index[i] for end in end_index[i]] for i in range(start_index.size(0))]
        logits = torch.Tensor([[(start_preds[i][d[0]]+end_preds[i][d[1]]-cls_logits[i]) if self.is_valid_index(examples[i], d) else -1e3 for d in indices[i]] for i in range(cls_logits.size(0))])
        logits, logit_indices = torch.max(logits, dim=1) 
        final_indices = [indices[i][logit_indices[i]] for i in range(cls_logits.size(0))]
        for i, example in enumerate(examples):
            if example.example_id not in self.best_scores.keys():
                self.best_scores[example.example_id] = 0.0
            if self.best_scores[example.example_id] < logits[i]:
#             and int(torch.max(class_preds,dim=1)[1][i])==0: # predict 'known'
                self.best_scores[example.example_id] = logits[i]
                self.examples[example.example_id] = example
                self.results[example.example_id] = [example.doc_start, final_indices[i]]
            #预测为无答案的样本
            if example.example_id not in self.results.keys():
                self.unknown_examples[example.example_id] = example
            elif example.example_id in self.unknown_examples.keys():
                self.unknown_examples.pop(example.example_id)
                

    def generate_predictions(self):
        """Generate predictions of each examples.
        """
        answers = []
        logger.info("*** generate predictions ***")
        logger.info("*** eval examples: {} ***".format(len(self.best_scores))) 
        logger.info("*** known examples: {} ***".format(len(self.results)))
        logger.info("*** unknown examples: {} ***".format(len(self.unknown_examples)))
        assert len(self.best_scores)==len(self.examples)+len(self.unknown_examples)
        for example_id in self.best_scores.keys():
            if example_id in self.results.keys() and example_id in self.examples.keys():
                doc_start, index = self.results[example_id]
                example = self.examples[example_id]
                tokenized_to_original_index = example.tokenized_to_original_index
                passage_token_start = doc_start+index[0]-example.question_len-2
                passage_token_end = doc_start + index[1]-example.question_len-2
                if passage_token_start >= len(tokenized_to_original_index)-1 or passage_token_start < 0:
                    start_index = -1
                else:
                    start_index = tokenized_to_original_index[passage_token_start]
                    
                if passage_token_end > len(tokenized_to_original_index)-1 or passage_token_end < 1:
                    end_index  = -1
                else:
                    end_index = tokenized_to_original_index[passage_token_end]
                assert start_index!=-1 and end_index!=-1
                answer = example.passage[start_index:end_index+1]
            else:
                answer = '疫情' #该样本经过预测没有答案
                example = self.unknown_examples[example_id]
            if not self.is_testing:
                answers.append({'example_id':example_id, 'pred':answer, 'label':example.passage[example.origin_start:example.origin_end+1]})
            else: 
                answers.append({'example_id':example_id, 'pred':answer, 'docid':example.doc_id})
        assert len(answers)==len(self.best_scores)
        return answers


    def score(self):
        data = self.generate_predictions()
        example_id = [d['example_id'] for d in data]
        prediction = [d['pred'] for d in data]
        if not self.is_testing:
            label = [d['label'] for d in data]
            df = pd.DataFrame({'example_id':example_id, 'prediction':prediction, 'label':label})
            hyps, refs = map(list, zip(*[[' '.join(list(d['pred'])), ' '.join(list(d['label']))] for d in data]))
            rouge = Rouge()
            scores = rouge.get_scores(refs, hyps, avg=True)
            return scores['rouge-l']['f'], df
        else:
            doc_id = [d['docid'] for d in data]
            df = pd.DataFrame({'id':example_id, 'docid':doc_id, 'answer':prediction})
            return df


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)