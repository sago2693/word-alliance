import torch
from evaluation import  test_model_multitask
import os
from multitask_classifier import get_args, seed_everything, MultitaskBERT
from evaluation import test_model_multitask

def test_best_model(path):
    saved = torch.load(path, map_location=torch.device('cuda'))
    config = saved['model_config']
    model = MultitaskBERT(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    os.makedirs(os.path.dirname("./predictions/prediction_test/"), exist_ok=True)
    os.makedirs(os.path.dirname("./predictions/prediction_evaluation/"), exist_ok=True)
    return model 

if __name__ == "__main__":

    path_model_list = os.listdir("./models/")
    model_path = os.path.join("./models/", path_model_list[0])
    args = get_args()
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    print(torch.cuda.is_available() )
    seed_everything(args.seed)  # fix the seed for reproducibility
    args.sts_test_out = f"{args.sts_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    args.para_test_out = f"{args.para_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    args.sst_test_out = f"{args.sst_test_out}-{args.option}-epoch-number-from-{args.epochs}-{args.lr}-model_batch_size_{args.batch_size}.csv"
    model = test_best_model( model_path)
    test_model_multitask(args, model, device)
            

    
    
