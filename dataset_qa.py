import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer

import json
import unicodedata

class QADataset(Dataset):
    
    def parse_data(self, path):              
        with open(path, 'r') as file:
            data_dict = json.load(file)        
        data = [p for d in data_dict['data'] for p in d['paragraphs']]
        if self.start is None and self.end is None:
            return data
        else:
            start = self.start or 0
            end = self.end or len(data)
            start = min(max(0,start),len(data))
            end = min(max(0,end),len(data))
            assert start<end, "Check input start and end."
            return data[start:end]
        
    
    def preprocess_text(self, inputs):
        """
        From Albert tokenizer, for making sure same text before and after tokenization.
        This is to ensure the accuracy of span index.
        """
        if self.tokenizer.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.tokenizer.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c) and not repr(c).startswith("'\\u")])
        if self.tokenizer.do_lower_case:
            outputs = outputs.lower()
        
        outputs = unicodedata.normalize("NFKC", outputs)
        # outputs = outputs.replace('\ufeff','')
        # outputs = outputs.replace('\u200e','')

        return outputs
    
    def normalize_data(self, data):
        for d in data:
            context_updt = self.preprocess_text(d['context'])
            if len(context_updt) == len(d['context']):
                continue # no change due to text preprocessing
            for q in d['qas']:
                if not q['is_impossible']:
                    for a in q['answers']:
                        a['answer_start'] = len(self.preprocess_text(re.sub('\s$','_',d['context'][:a['answer_start']])))
                        a['text'] = self.preprocess_text(a['text'])
            d['context'] = context_updt
        return data 

    def make_dataset(self, paths):
        data = [self.parse_data(p) for p in paths]
        data = [di for d in data for di in d]
        data = self.normalize_data(data)
        return data
        
    def __init__(self, paths, tokenizer, start=None, end=None, random_state=None, training=True, **kwargs):

        if isinstance(paths, str):
            paths = [paths]
        
        self.tokenizer = tokenizer  
        assert self.tokenizer.__class__ == AlbertTokenizer, "Current context span index to tokenized span index conversion is specific to AlbertTokenizer."
        self.start = start
        self.end = end
        self.random_state = random_state
        self.training = training
        self.data = self.make_dataset(paths)
        self.idxs = [(i,j) for i, p in enumerate(self.data) for j, q in enumerate(p['qas'])]
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        
        random.seed(self.random_state)
                
        pid, qid = self.idxs[idx]
        paragraph = self.data[pid]
        
        context = paragraph['context']
        context = self.tokenizer.tokenize(context)
        token_len_list = [0]+[len(t) for i, t in enumerate(context)]
        token_len_cumsum = np.cumsum(np.array(token_len_list))
        context_ids = self.tokenizer.convert_tokens_to_ids(context)
        
        unk = paragraph['qas'][qid]['is_impossible']
        if not unk:
            answers = paragraph['qas'][qid]['answers']
            if self.training:
                answer_idx = random.randrange(len(answers))
                span = (answers[answer_idx]['answer_start'],len(answers[answer_idx]['text']))
                token_span = self.get_token_span(token_len_cumsum, span)
            else:
                span = [(a['answer_start'],len(a['text'])) for a in answers]
                token_span = [self.get_token_span(token_len_cumsum, s) for s in span]
        else:
            token_span = None      
        
        question = paragraph['qas'][qid]['question']
        question = self.tokenizer.tokenize(question)
        question_ids = self.tokenizer.convert_tokens_to_ids(question)
        
        q_idx = paragraph['qas'][qid]['id']
        return (question_ids, unk, token_span, context_ids, q_idx)
    
    @staticmethod
    def get_token_span(token_len_cumsum, span):
        """
        returned start token points to the first token of the span
        returned end token points to the last token of the span
        """
        start_token = int(np.argwhere(token_len_cumsum<=span[0])[-1][0])
        end_token = int(np.argwhere(token_len_cumsum>=sum(span)+1)[0][0]) # plus 1 for the additional '_'
        token_span = (start_token, max(0,end_token-1)) 
        return token_span

class QADataLoader(object):
    
    def __init__(self, dataset, max_len, batch_size, n_jobs=0, shuffle=False, **kwargs):
        self.dataset = dataset
        self.max_len= max_len
        self.training = dataset.training
        self.tokenizer = dataset.tokenizer
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                     num_workers=n_jobs, collate_fn=self.collate_func)
    
    def get_dataloader(self):
        return self.dataloader
    
    def complete_sent(self,raw_sent):
        return [self.tokenizer.bos_token_id]+raw_sent+[self.tokenizer.eos_token_id]
    
    def test(self):
        self.dataset.training = False
        self.training = False
        
    def train(self):
        self.dataset.training = True
        self.training = True
    
    def collate_func(self, data):
                
        questions, unk, token_span, context, q_idx = zip(*data)
        unk = list(unk)
        input_ids = []
        token_type_ids = []
        for i, q in enumerate(questions):
            c_ids = self.complete_sent(context[i])
            q_ids = self.complete_sent(q)
            if len(c_ids)+len(q_ids)>self.max_len:
                # TODO: fix by preliminary match
                # Notet that the cut by [-self.max_len:] changes start/end index -> can't use.
                unk[i] = True 
            input_ids.append(torch.tensor((c_ids+q_ids)[-self.max_len:], dtype=torch.long))
            token_type_ids.append(torch.cat([torch.zeros(len(c_ids), dtype=torch.long), torch.ones(len(q_ids), dtype=torch.long)],axis=-1)[-self.max_len:])
        output ={ 'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                 ,'token_type_ids': pad_sequence(token_type_ids, batch_first=True, padding_value=1)}
        output['attention_mask'] = 1*(output['input_ids']!=self.tokenizer.pad_token_id)
        
        if self.training:
            start_positions = []
            end_positions = []
            for i, t in enumerate(token_span):
                if not unk[i]:
                    start_positions.append(t[0]+1) # +1 for the bos token
                    end_positions.append(t[1]+1) # +1 for the bos token
                else:
                    #start_positions.append(len(context[i])+1) # point to eos if no answer
                    #end_positions.append(len(context[i])+1) # point to eos if no answer
                    start_positions.append(-1)
                    end_positions.append(-1)
            output['start_positions'] = torch.tensor(start_positions, dtype=torch.long)
            output['end_positions'] = torch.tensor(end_positions, dtype=torch.long)
        else:
            positions = []
            for i, t in enumerate(token_span):
                if not unk[i]:
                    positions.append([(ti[0]+1,ti[1]+1) for ti in t]) # +1 for the bos token
                else:
                    positions.append(-1)
            output['positions'] = positions
        
        output['q_idx'] = q_idx
        return output
