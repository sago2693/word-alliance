# It is better to copy the code here instead of importing to prevent the arg part from running
import torch

from torch.utils.data import DataLoader
import time, random, numpy as np, argparse
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from types import SimpleNamespace
from tokenizers.processors import TemplateProcessing

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data
from bert import BertModel
from data_loader import MultiTaskBatchSampler,MultiTaskDataset
from optimizer import AdamW

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, compute_loss_weights
from tokenizer import BertTokenizer

import os

N_SENTIMENT_CLASSES = 5
TQDM_DISABLE=False


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=config.local_files_only)
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.drop = torch.nn.Dropout(p=0.3)
        self.sst_classifier = torch.nn.Linear(self.bert.config.hidden_size, N_SENTIMENT_CLASSES)
        self.para_classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.sts_classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=config.local_files_only)

    def forward(self, input_ids, attention_mask,token_type_ids):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        bert_out = self.bert(input_ids, attention_mask,token_type_ids) 
        dropped = self.drop(bert_out['pooler_output'])
        return dropped

    def predict(self,input_ids,attention_mask,token_type_ids,task_id):
        cls_hidden_state = self.forward(input_ids, attention_mask,token_type_ids)

        if task_id==0:
            return self.sst_classifier(cls_hidden_state)
        elif task_id==1:
            return self.para_classifier(cls_hidden_state)
        elif task_id==2:
            return self.sts_classifier(cls_hidden_state)
        else:
            raise ValueError("Invalid task_id value. Expected 0, 1, or 2.")



def save_model(model, optimizer, args, config, filepath,epoch, batch_size, weighted_avg,  dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr,loss):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(999999999999999999999)
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    model_number = 1
    print(batch_size)
    # while os.path.exists(os.path.join(filepath, f"{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{batch_size}.pt")):
        # model_number += 1

    model_path = os.path.join(filepath,  f"{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{batch_size}.pt")
    torch.save(save_info, model_path)

    os.makedirs(os.path.dirname("./Models_Meta_Data/"), exist_ok=True)
    txt_filename = os.path.join("./Models_Meta_Data/", f"{args.option}-epoch-number {epoch}-from-{args.epochs}-{args.lr}-model_{model_number}.txt")
    # txt_filename = os.path.splitext(txt_path)[0] + ".txt"
    with open(txt_filename, 'w') as txt_file:
        txt_file.write(f"Model {model_number} information:\n")
        txt_file.write(f"weighted_avg: {weighted_avg}\n")
        txt_file.write(f"dev_sentiment_accuracy: {dev_sentiment_accuracy}\n")
        txt_file.write(f"dev_paraphrase_accuracy: {dev_paraphrase_accuracy}\n")
        txt_file.write(f"dev_sts_corr: {dev_sts_corr}\n")
        txt_file.write(f"Loss: {loss}\n")
        txt_file.write(f"Epoch: {epoch}\n")
        txt_file.write(f"from total Epochs: {args.epochs}\n")

        txt_file.write(f"Learning rate: {args.lr}\n")
        txt_file.write(f"Batch size: {batch_size}\n")
    print(f"save the model to {filepath}")

#Collate function dependent on current task
class CustomCollateFn:
    def __init__(self, collate_fns):
        self.collate_fns = collate_fns

    def __call__(self, batch):
        task_id,_= batch[0] #This tuple is defined in the MultiTaskDataset class
        #This only works if a batch only contains data from one task
        collate_fn = self.collate_fns[task_id]
        actual_batch = [actual_batch for _, actual_batch in batch]
        return collate_fn(actual_batch)

def train_multitask(args):
        
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train') #Itis correct to use this slit for dev. The other option is test which does not load the labels

    #Sentiment analysis
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)



    if torch.cuda.is_available():
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn, pin_memory=True )

        #Paraphrasing
        paraphrase_train_data = SentencePairDataset(para_train_data, args, isRegression =False)
        paraphrase_dev_data = SentencePairDataset(para_dev_data, args, isRegression =False)

        paraphrase_dev_dataloader = DataLoader(paraphrase_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=paraphrase_dev_data.collate_fn,  pin_memory=True)

        #sts
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression =True)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression =True)

        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn,  pin_memory=True)

        #MTL data loader
        train_datasets = [sst_train_data,paraphrase_train_data, sts_train_data]
        #Temporarily initialized here but later in epoch loop to update current epoch and do annealed sampling
        mtl_sampler = MultiTaskBatchSampler(        datasets=train_datasets,
            current_epoch=1,
            total_epochs=args.epochs,
            batch_size = args.batch_size,
            mix_opt=1,
            extra_task_ratio=0,
            bin_size=64,
            bin_on=False,
            bin_grow_ratio=0.5,
            sampling='sequential')

        multi_task_train_dataset = MultiTaskDataset(train_datasets)

        collate_fns = {
            0: sst_train_data.collate_fn,
            1: paraphrase_train_data.collate_fn,
            2: sts_train_data.collate_fn
        }

        # Creating the custom collate function using the dictionary of collate functions
        # Linked to each task id
        custom_collate_fn = CustomCollateFn(collate_fns)

        multi_task_train_data = DataLoader(
        multi_task_train_dataset,
        batch_sampler=mtl_sampler,
        collate_fn = custom_collate_fn,
        pin_memory=True
        )
    else:
        sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
        sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train') #Itis correct to use this slit for dev. The other option is test which does not load the labels

        #Sentiment analysis
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        #Paraphrasing
        paraphrase_train_data = SentencePairDataset(para_train_data, args, isRegression =False)
        paraphrase_dev_data = SentencePairDataset(para_dev_data, args, isRegression =False)

        paraphrase_dev_dataloader = DataLoader(paraphrase_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=paraphrase_dev_data.collate_fn)

        #sts
        sts_train_data = SentencePairDataset(sts_train_data, args, isRegression =True)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression =True)

        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

        #MTL data loader
        train_datasets = [sst_train_data,paraphrase_train_data, sts_train_data]
        #Temporarily initialized here but later in epoch loop to update current epoch and do annealed sampling
        mtl_sampler = MultiTaskBatchSampler(        datasets=train_datasets,
            current_epoch=1,
            total_epochs=args.epochs,
            batch_size = args.batch_size,
            mix_opt=1,
            extra_task_ratio=0,
            bin_size=64,
            bin_on=False,
            bin_grow_ratio=0.5,
            sampling='sequential')

        multi_task_train_dataset = MultiTaskDataset(train_datasets)

        collate_fns = {
            0: sst_train_data.collate_fn,
            1: paraphrase_train_data.collate_fn,
            2: sts_train_data.collate_fn
        }

        # Creating the custom collate function using the dictionary of collate functions
        # Linked to each task id
        custom_collate_fn = CustomCollateFn(collate_fns)

        multi_task_train_data = DataLoader(
        multi_task_train_dataset,
        batch_sampler=mtl_sampler,
        collate_fn = custom_collate_fn
        )


    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option,
              'local_files_only': args.local_files_only}
    print("12")

    config = SimpleNamespace(**config)
    print("13")
    model = MultitaskBERT(config)
    print("14")
    

    model = model.to("cuda")
    print("15")

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    print("16")
    best_metric = 0.2
    print(torch.cuda.is_available())
    print(f"running the train on the {device}")
    # Run for the specified number of epochs
        
    for epoch in range(args.epochs):
        print("17")
        model.train()
        train_loss = 0
        num_batches = 0
        sst_train_loss_list = []
        paraphrase_train_loss_list = []
        sts_train_loss_list = []

        for batch in tqdm(multi_task_train_data, desc=f'train-{epoch}', disable=TQDM_DISABLE):

            #Batch loading, prediction and loss depending on task:
            

            optimizer.zero_grad()
            b_task_id, b_ids, b_mask, b_token_type_ids, b_labels = (
            batch['task_id'],
            batch['token_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
            batch['labels'].to(device))
            
            
            logits = model.predict(input_ids=b_ids,attention_mask=b_mask,token_type_ids=b_token_type_ids,task_id=b_task_id)
            batch_loss = [0]*3
            if b_task_id==0: #Sentiment analysis
                sst_loss = F.cross_entropy(logits, b_labels.view(-1), reduction='mean')
                batch_loss[b_task_id]=sst_loss
                sst_train_loss_list.append(sst_loss.item()) #value, not tensor

            elif b_task_id==1: #Paraphrasing
                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                paraphrase_loss=bce_loss(logits.view(-1),b_labels.to(torch.float64)) #Change these logits
                batch_loss[b_task_id]=paraphrase_loss
                paraphrase_train_loss_list.append(paraphrase_loss.item())

            elif b_task_id==2: # Text similarity

                sigmoid = nn.Sigmoid()
                probabilities = sigmoid(logits) #maps logits to range 0 to 1
                # Define the MSE loss function
                mse_loss = nn.MSELoss(reduction='mean')
                b_labels_scaled = (b_labels / 5).float() #Divide between 5 to match range 0 to 1 of logit. Float required due to loss calculation error
                sts_loss = mse_loss(probabilities.view(-1), b_labels_scaled)
                batch_loss[b_task_id]=sts_loss
                sts_train_loss_list.append(sts_loss.item())
                        
            else:
                raise ValueError("Invalid b_task_id value. Expected 0, 1, or 2.")
            

            losses_list = [sst_train_loss_list,paraphrase_train_loss_list,sts_train_loss_list]
            #Compute weighted loss
            weights = compute_loss_weights(losses_list)

            total_loss = 0
            for loss, weight in zip(batch_loss,weights):
                total_loss+=loss*weight
                
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1





            #End of training batches
        print("inside eval ")
        #Start dev evaluation 
        (dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids,
            dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids,
            dev_sts_corr, dev_sts_y_pred, dev__sent_ids) = model_eval_multitask(sst_dev_dataloader,
                                                                        paraphrase_dev_dataloader,sts_dev_dataloader,model, device  )
        print("outside eval ")
        #We have to weight or average the three sores to save the best model.
        # In the diven code only sst is used

        weighted_avg = 0.333 * dev_sentiment_accuracy + 0.333 * dev_paraphrase_accuracy + 0.333 * ((dev_sts_corr +1) / 2)
        print(f"the weighted avg {weighted_avg}")
        if   weighted_avg >=  best_metric :
            best_metric = weighted_avg
            save_model(model, optimizer, args, config, args.filepath, epoch,args.batch_size, weighted_avg,  dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr, train_loss)
            print("model saved")

        print(f"Epoch {epoch}: train loss : {train_loss :.3f}, dev paraphrase acc : {dev_paraphrase_accuracy :.3f}, dev sentiment acc : {dev_sentiment_accuracy :.3f}, dev sts corr : {dev_sts_corr :.3f}, Best Metric : {best_metric :.3f}")
        #Add train metrics to this print
    


def test_model(args, path ):
    with torch.no_grad():
        device = torch.device('cuda') if True else torch.device('cpu')
        # saved = torch.load(args.filepath)
        ###TODO change the file path to the one with the best peformance in terms of the best metric
        
        saved = torch.load(path)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--local_files_only", action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    
    device = torch.device('cuda') if True else torch.device('cpu')
    print(f"before the args {torch.cuda.is_available()}")
    args = get_args()
    args.use_gpu = True
    print(args.batch_size)
    args.filepath = f'./models/' # save path
    print(f"after the args {torch.cuda.is_available()}")
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    args.sts_test_out = f"{args.sts_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    args.para_test_out = f"{args.para_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    args.sst_test_out = f"{args.sst_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    
    model_path = os.path.join( args.filepath,  f"{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.pt")
    test_model(args, model_path)

