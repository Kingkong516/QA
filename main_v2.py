import os
import random
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.optim import Adam
import logging
import json

path = r'U:\Learning\NLP\NLG\QA'
os.chdir(path)

from model_qa import QAModel
from dataset_qa import QADataset, QADataLoader
from evaluate import QAEvaluator
from trainer import CLR, LRLinearDecrease, LRConst, LRRangeTest

############################## SET UP #########################################
data_path = r'U:\Learning\NLP\NLG\QA\datasets\SQUAD2'
data_path_dict = {'train': os.path.join(data_path,'train-v2.0.json')
                  ,'dev': os.path.join(data_path,'dev-v2.0.json')
                  ,'pred':os.path.join(data_path,'dev-evaluate-v2.0-in1.json')}

logging.basicConfig(filename='QA_LOG.txt',level=logging.INFO,
                    format='%(asctime)s %(levelname)s : %(message)s')
sanity = False

if sanity:
    config = {
        'top': 10
        ,'random_state':0
        ,'batch_size' : 10
        ,'n_epochs': 200
        ,'lr': 1e-4
        ,'decay_niter': 10
        ,'lr_decay': 0.8
        ,'name': 'sanity_qa'
        ,'model':None
    }
else:
    config = {
        'top': None
        ,'batch_size' : 10
        ,'n_epochs': 10
        ,'lr': 1e-3
        ,'decay_niter': 100
        ,'lr_decay': 0.95
        ,'name': 'qa_train_full_2x_unk_loss'
        ,'model':'model_qa_train_full_2x_unk_loss_202011140847'
    }

logging.info("************************ STARTING *****************************")
logging.info(config)

model = QAModel(cuda=torch.cuda.is_available())
if config['model'] is not None:
    model.load_state_dict(torch.load(f".\checkpoints\{config['model']}"
                                     ,map_location=torch.device(model.device)))    

############################## TRAIN ##########################################
train_start = 1000
dev_size = 100
dataset = QADataset(data_path_dict['train'], model.tokenizer, start=train_start, end=18000, **config) # end=18000
dataloader = QADataLoader(dataset,model.max_len,**config)

n_steps = config['n_epochs']*len(dataloader.dataloader)
# lr_scheduler = LRRangeTest(n_steps=n_steps,lb=1e-6,ub=1e-3)
# lr_scheduler = LRConst(n_steps=n_steps,lr=1e-4)
# lr_scheduler = LRLinearDecrease(n_steps=n_steps,lb=2e-5,ub=2e-4)
lr_scheduler = CLR(n_steps=n_steps,lb=2e-7,ub=2e-6,reserve=0.1) # for training full model
# lr_scheduler = CLR(n_steps=n_steps,lb=1e-4,ub=1e-3,reserve=0.1) # for training unk and qa head only
lr_generator = lr_scheduler.get_lr() 
n_chkpt = 10
n_dev_loss = 100*config['n_epochs']
n_iter_per_chkpt = int(np.floor(n_steps/n_chkpt))
n_iter_per_dev_loss = int(np.floor(n_steps/n_dev_loss))

optimizer = Adam([{'params': model.model.parameters()}], lr=config['lr'])
# optimizer = Adam(list(model.model.unk_outputs.parameters())\
#                 +list(model.model.qa_outputs.parameters())
#                 , lr=0.001)
#if config['model'] is not None:
#    optimizer_name = config['model'].replace('model','optimizer')
#    optimizer.load_state_dict(torch.load(f".\checkpoints\{optimizer_name}"
#                                         ,map_location=torch.device(model.device)))

random.seed(0)
model.train()
evaluator = QAEvaluator()
start = time.time()
n_update = 0
n_update_test = 0
losses = []
losses_test = []
loss_train = 0
loss_test = 0
for e in range(0,config['n_epochs']): 
    n_iter = len(dataloader.dataloader)
    for i, sample in enumerate(tqdm(dataloader.dataloader)): 
        
        n_update += 1
        
        # Update learning rate
        lr = next(lr_generator)
        logging.info(f"""LEARNING RATE: {lr}""")
        for g in optimizer.param_groups:
            g['lr'] = lr 
        
        # Update model and output loss
        optimizer.zero_grad()
        output = model(sample)
        loss = output.loss
        loss.backward()
        optimizer.step()
        if model.cuda:
            del output, sample
            torch.cuda.empty_cache()
        loss_train += loss.item()
        losses.append(loss_train/n_update)
        
        # output summary
        ts = datetime.now().strftime("%Y%m%d%H%M")
        if (n_update<=30) or \
           (n_update%n_iter_per_dev_loss==0) or \
           (n_update==n_steps):  
            n_update_test += 1
            dev_start = random.randrange(train_start-dev_size)
            model.eval()
            dataset_dev = QADataset(data_path_dict['train'], model.tokenizer, start=dev_start, end=dev_start+dev_size, **config)
            dataloader_dev = QADataLoader(dataset_dev,model.max_len,**config)
            loss_test += evaluator.get_loss(model, dataloader_dev)
            losses_test.append((n_update,loss_test/n_update_test))
            model.train()
            result = pd.DataFrame({'train_loss':losses},index = range(1,n_update+1))
            df_dev = pd.DataFrame(losses_test, columns=['iteration','test_loss']).set_index('iteration')
            result = pd.merge(result,df_dev,left_index=True,right_index=True,how='left')
            sns.set(rc={'figure.figsize':(11.7,8.27)})
            fig = sns.lineplot(data=result).set_title('loss').get_figure()
            fig.savefig(f"{ts}_LOSS_epoch_{e+1}_iter_{i+1}_{config['name']}.png", dpi=300)
            fig.clf()
            
        # Checkpoint and log
        if (n_update%n_iter_per_chkpt==0) or (n_update==n_steps):
            elapsed = round((time.time()-start)/60,2)
            logging.info(f"""ELAPSED: {elapsed} MIN; EPOCH: {e+1}/{config['n_epochs']}; ITERATION: {i+1}/{n_iter}; LOSS: {losses[-1]}""")
            torch.save(model.state_dict(), f".\checkpoints\model_{config['name']}_{ts}")
            torch.save(optimizer.state_dict(), f".\checkpoints\optimizer_{config['name']}_{ts}")

############################## EVALUATE #######################################
dataloader.test()
evaluator = QAEvaluator()
eval_result = evaluator.evaluate(model, dataloader)
if sanity:        
    if eval_result['accuracy'] >0.99999:
        print("PASSED SANITY CHECK!")
else:
    dataset_test = QADataset(data_path_dict['dev'], model.tokenizer, **config)
    dataloader_test = QADataLoader(dataset_test,model.max_len,**config)
    dataloader_test.test()
    eval_result_test = evaluator.evaluate(model, dataloader_test)
    print(eval_result)
    print(eval_result_test)
    
sample = dataloader_test.collate_func([dataset_test[i] for i in range(5)])

####################### OFFICIAL EVALUATE #####################################
dataset_test = QADataset(data_path_dict['dev'], model.tokenizer, **config)
dataloader_test = QADataLoader(dataset_test,model.max_len,**config)
evaluator = QAEvaluator()
model.eval()
predicted_text = evaluator.output_prediction(model, dataloader_test)
file_nm = os.path.join(data_path,f"dev-evaluate-v2.0_{config['model']}.json")
with open(file_nm, 'w') as fp:
    json.dump(predicted_text, fp)

