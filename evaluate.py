import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class QAEvaluator(object):
    
    def __init__(self):
        pass
    
    def get_loss(self, model, dataloader):
        """
        This is to get the output of the loss function.
        """
        self.model = model
        self.dataloader = dataloader
        losses = []
        for i, sample in enumerate(self.dataloader.dataloader):
            output = model(sample)
            losses.append(output.loss.item())
            if self.model.cuda:
                del sample, output
                torch.cuda.empty_cache()
        return sum(losses)/len(losses)  

    def output_prediction(self, model, dataloader, unk_token=-1):
        self.model = model
        self.dataloader = dataloader
        
        print('\nRUNNING PREDICTION FOR EVALUATION...')   
        predicted_text = {}
        for i, sample in enumerate(tqdm(self.dataloader.dataloader)):
            prediction = self.model.predict(sample)
            for i, q_idx in enumerate(sample['q_idx']):
                start_pred = prediction[0][i].item()
                end_pred = prediction[1][i].item()
                if start_pred == unk_token:
                    predicted_text[q_idx] = ''
                else:
                    text_ids = sample['input_ids'][i][start_pred:end_pred+1]
                    text = self.model.tokenizer.convert_ids_to_tokens(text_ids)
                    text = ''.join(text).replace('▁',' ').strip()
                    predicted_text[q_idx] = text
                        
            if self.model.cuda:
                del sample, prediction
                torch.cuda.empty_cache()  
        return predicted_text                  
            
    def evaluate(self, model, dataloader):
        """
        This is to get 
        (1) precision and recall of unk
        (2) accuracy of start and end among those correctly identified as unk.
        """
        
        self.model = model
        self.dataloader = dataloader
        self.test_mode = not dataloader.training
        
        print('\nRUNNING EVALUATOR...')
        if self.test_mode:
            ground_truth = []
            prediction = []            
            for i, sample in enumerate(tqdm(self.dataloader.dataloader)):
                ground_truth.extend(sample['positions'])
                prediction.append(self.model.predict(sample))
                if self.model.cuda:
                    del sample
                    torch.cuda.empty_cache() 
            start_pred, end_pred = zip(*prediction)
            start_pred = torch.cat(start_pred,axis=-1).cpu().numpy()
            end_pred = torch.cat(end_pred,axis=-1).cpu().numpy()
            self.test_data = (ground_truth, start_pred, end_pred)
            eval_result = self.get_qa_performance_test(ground_truth, start_pred, end_pred)
        else:
            ground_truth = []
            prediction = []            
            for i, sample in enumerate(tqdm(self.dataloader.dataloader)):
                ground_truth.append((sample['start_positions'],sample['end_positions']))
                prediction.append(self.model.predict(sample))
                if self.model.cuda:
                    del sample
                    torch.cuda.empty_cache() 
            start, end = zip(*ground_truth)
            start_pred, end_pred = zip(*prediction)
            start_true = torch.cat(start,axis=-1).cpu().numpy()
            start_pred = torch.cat(start_pred,axis=-1).cpu().numpy()
            end_true = torch.cat(end,axis=-1).cpu().numpy()
            end_pred = torch.cat(end_pred,axis=-1).cpu().numpy()
            eval_result = self.get_qa_performance(start_true, start_pred, end_true, end_pred)
            
        return eval_result
    
    @staticmethod
    def get_qa_performance_test(y, start_pred, end_pred, unk_token=-1):
        """
        Allow multiple spans as correct answer, testing style.
        The accuracy score is out of those predicted as known.
        """
        
        result = {}
        
        # unk precision and recall
        unk_true = np.array([1 if i==unk_token else 0 for i in y])
        unk_pred = 1*(start_pred==unk_token)
        prec, recall, _, _ = precision_recall_fscore_support(unk_true, unk_pred, labels=[1])
        result['unk_precision'] = prec[0]
        result['unk_recall'] = recall[0]
        
        # start accuracy
        start_true = [[j[0] for j in i] if isinstance(i,list) else [i] for i in y]
        start_accu = [s in start_true[i] for i,s in enumerate(start_pred) if s!= unk_token]
        
        # end accuracy
        end_true = [[j[1] for j in i] if isinstance(i,list) else [i] for i in y]
        end_accu = [s in end_true[i] for i,s in enumerate(end_pred) if s!= unk_token]
        
        # start and end accuracy
        accu = [se in (y[i] if isinstance(y[i],list) else [y[i]]) 
                for i,se in enumerate(zip(start_pred,end_pred)) if se[0]!= unk_token]
        
        result['start_accuracy'] = np.array(start_accu).mean()
        result['end_accuracy'] = np.array(end_accu).mean()
        result['accuracy'] = np.array(accu).mean()
        
        return result
        
    @staticmethod            
    def get_qa_performance(start_true, start_pred, end_true, end_pred, unk_token=-1):
        """
        Allow only 1 correct span, training style.
        The accuracy score is out of those predicted as known.
        """
        
        result = {}
        
        # unk precision and recall
        unk_true = 1*(start_true==unk_token)
        unk_pred = 1*(start_pred==unk_token)
        prec, recall, _, _ = precision_recall_fscore_support(unk_true, unk_pred, labels=[1])
        result['unk_precision'] = prec[0]
        result['unk_recall'] = recall[0]
        
        # start accuracy
        k = (start_pred!=unk_token)
        result['start_accuracy'] = accuracy_score(start_true[k],start_pred[k])
        
        # end accuracy
        result['end_accuracy'] = accuracy_score(end_true[k],end_pred[k])
        
        # start and end accuracy
        result['accuracy'] = ((start_true[k]==start_pred[k])&(end_true[k]==end_pred[k])).sum()/k.sum()
        
        return result