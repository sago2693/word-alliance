import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from data_loader import MultiTaskBatchSampler,MultiTaskDataset
from optimizer import AdamW
from tqdm import tqdm


from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, compute_total_loss


TQDM_DISABLE=True

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
N_SENTIMENT_CLASSES = 5


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


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        bert_out = self.bert(input_ids, attention_mask) 
        dropped = self.drop(bert_out['pooler_output'])
        out = self.linear(dropped)
        sentence_embeddings = out.last_hidden_state[:, 0, :]
        return sentence_embeddings


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        hidden_states = self.forward(input_ids, attention_mask)
        # second dropout might be overkill causing over-regularization
        # hidden_states = self.drop(hidden_states)
        logits = self.sst_classifier(hidden_states[:, 0, :])
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''

        ### TODO
        hidden_states_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_states_2 = self.forward(input_ids_2, attention_mask_2)
        combined_hidden_states = torch.cat((hidden_states_1[:, 0 :], hidden_states_2[:, 0, :]), dim=-1)
        logits = self.para_classifier(combined_hidden_states)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        hidden_states_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_states_2 = self.forward(input_ids_2, attention_mask_2)
        # Absolute difference is more efficient and reduces dimensionality. Might not perform as well though.
        # torch.abs(hidden_states_1[:, 0, :] - hidden_states_2[:, 0, :])
        # logits = self.sts_classifier(combined_rep)
        combined_hidden_states = torch.cat((hidden_states_1[:, 0 :], hidden_states_2[:, 0, :]), dim=-1)
        logits = self.sts_classifier(combined_hidden_states)
        return logits




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
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
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
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
        total_epochs=5,
        batch_size = 128,
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

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc_sst = 0
    best_dev_acc_paraphrase = 0
    best_dev_corr_sts = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        sst_train_loss_list = []
        paraphrase_train_loss_list = []
        sts_train_loss_list = []
        #TODO IMPLEMENT A SIMPLE SEQUENTIAL DATALOADER TO TEST
        for batch in tqdm(multi_task_train_data, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            #TODO Here we have to separate tokens according to the task.
            #TODO I will have to return the dataset_id in the collate method to 
            #get it from the same dataloader. Once with the dataset_id I can choose which method and loss to use

            b_task_id = batch['task_id']

            #Batch loading, prediction and loss depending on task:
            optimizer.zero_grad()

            if b_task_id==0: #Sentiment analysis
                
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                logits_sst = model.predict_sentiment(b_ids, b_mask)
                sst_loss = F.cross_entropy(logits_sst, b_labels.view(-1), reduction='mean')
                sst_train_loss_list.append(sst_loss)

            elif b_task_id==1: #Paraphrasing

                b_ids_1, b_mask_1,b_ids_2,b_mask_2, b_labels = (batch['token_ids_1'],
                        batch['attention_mask_1'], batch['token_ids_2'],
                        batch['attention_mask_2'], batch['labels'])
                
                logits_paraphrase = model.predict_paraphrase(b_ids_1,
                            b_mask_1,
                           b_ids_2, 
                           b_mask_2)
                #TODO. THESE VALUES must be checked
                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                paraphrase_loss=bce_loss(logits_paraphrase, b_labels.view(-1)) #Change these logits
                paraphrase_train_loss_list.append(paraphrase_loss)

            elif b_task_id==2: # Text similarity
                logits_sts = model.predict_similarity(b_ids, b_mask)
                
                #Complete paraphrase loss and sts loss. 

                #TODO. THESE VALUES must be checked
                # Apply sigmoid activation to logits
                sigmoid = nn.Sigmoid()
                probabilities = sigmoid(logits_sts) #maps logits to range 0 to 1
                # Define the MSE loss function
                mse_loss = nn.MSELoss(reduction='mean')
                sts_loss = mse_loss(probabilities, b_labels.view(-1))
                sts_train_loss_list.append(sts_loss)

            losses_list = [sst_train_loss_list,paraphrase_train_loss_list,sts_train_loss_list]
            #Compute weighted loss
            loss,variances = compute_total_loss(losses_list)
            print("list of performance variance by training",variances)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            #End of training batches

        #Start dev evaluation 
        (dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids,
         dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids,
           dev_sts_corr, dev_sts_y_pred, dev__sent_ids) = model_eval_multitask(sst_dev_dataloader,
                                                                      paraphrase_dev_dataloader,sts_dev_dataloader,model, model.device  )
        
        #We have to weight or average the three sores to save the best model.
        # In the diven code only sst is used


        if dev_sentiment_accuracy > best_dev_acc_sst and dev_paraphrase_accuracy >best_dev_acc_paraphrase and dev_sts_corr>best_dev_corr_sts:
            best_dev_acc_sst = dev_sentiment_accuracy
            best_dev_acc_paraphrase = dev_paraphrase_accuracy
            best_dev_corr_sts = dev_sts_corr
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev sentiment acc :: {dev_paraphrase_accuracy :.3f}, dev sentiment acc :: {dev_paraphrase_accuracy :.3f}, dev sts corr :: {best_dev_corr_sts :.3f}")
        #Add train metrics to this print



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
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
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
    parser.add_argument("--local_files_only", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
