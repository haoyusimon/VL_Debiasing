import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from clip_debiasing.datasets import FairFaceDebiasing_Gender, FairFaceDebiasing_Age, FairFaceDebiasing_Race, UTKfaceDebiasing_Gender, UTKfaceDebiasing_Age, UTKfaceDebiasing_Race, FairFaceDebiasing_Joint
import clip
from clip_debiasing.models.model_vl_debiasing import DebiasedCLIP
# from clip_debiasing.models.model_vl_debiasing_h_14 import DebiasedCLIP # for ViT-H/14

import utils
import eval_train
import argparse
import os
import sys
from eval_all import run_eval


def train(model, data_loader, optimizer, epoch=None, warmup_steps=None, device=None, scheduler=None, config=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = batch["img"].to(device)
        text1, text2 = torch.squeeze(batch["text1"].to(device)), torch.squeeze(batch["text2"].to(device))
        loss = model(image, text1, text2, epoch)

        loss.backward()

        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])   
        sys.stdout.flush() # timely update the logs
        sys.stderr.flush()

def main(args=None, config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="latest", type=str)
    args = parser.parse_args()

    # create version directory and copy model file 
    work_dir = os.path.join('.',f'exp/{args.version}')
    os.makedirs(work_dir)

    # to change
    dataset_name = 'fairface'
    # dataset_name = 'utkface'

    attribute = 'gender'
    # attribute = 'age'
    # attribute = 'race'

    backbone = 'ViT-B/16'
    # backbone = 'ViT-B/32'
    # backbone = 'ViT-L/14'
    # backbone = ('ViT-H-14', 'laion2b_s32b_b79k')

    mlp1_hidden_size = 512

    mlp2_hidden_size = 1024

    alpha = 0.3
    lr = 5e-6


    print(mlp1_hidden_size, mlp2_hidden_size, alpha)
    config = f"Exp_{mlp1_hidden_size}_{mlp2_hidden_size}_{alpha}_{lr}"
    exp_dir = os.path.join('.',f'exp/{args.version}/{config}')
    os.makedirs(exp_dir)
    sys.stdout = open(os.path.join(exp_dir, 'stdout.log'), 'w')
    sys.stderr = open(os.path.join(exp_dir, 'stderr.log'), 'w')

    print(f"Experiment version: {args.version}, config: {config}, dataset: {dataset_name}, attribute: {attribute}, backbone: {backbone}")
            
    device = torch.device('cuda')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    epochs = 100
    model = DebiasedCLIP(backbone, device=device, mlp1_hidden_size=mlp1_hidden_size, mlp2_hidden_size=mlp2_hidden_size, alpha=alpha).to(device)
    if dataset_name == 'fairface':
        dataset = FairFaceDebiasing_Gender(tokenizer=clip.tokenize, transforms=model.preprocess) if attribute == 'gender' else FairFaceDebiasing_Age(tokenizer=clip.tokenize, transforms=model.preprocess) if attribute == 'age' else FairFaceDebiasing_Race(tokenizer=clip.tokenize, transforms=model.preprocess)
        # dataset = FairFaceDebiasing_Joint(tokenizer=clip.tokenize, transforms=model.preprocess) # for universally removing all types of biases
    else:
        dataset = UTKfaceDebiasing_Gender(tokenizer=clip.tokenize, transforms=model.preprocess) if attribute == 'gender' else UTKfaceDebiasing_Age(tokenizer=clip.tokenize, transforms=model.preprocess) if attribute == 'age' else UTKfaceDebiasing_Race(tokenizer=clip.tokenize, transforms=model.preprocess) 

    data_loader = DataLoader(dataset, batch_size=512, num_workers=4)
    
    optimizer = torch.optim.Adam(list(model.mlp1.parameters()) + list(model.mlp2.parameters()), 
                                lr=lr)

    best_details = {"epoch": 0, "eval": -100}
    count = 0
    for epoch in range(epochs):
        if count >= 10: 
            print(f"Early-stopped after {epoch} epochs.")
            break # early stop

        train(model, data_loader, optimizer, device=device, epoch=epoch)

        if (epoch+1) % 2 == 0:
            print(f"Evaluation for Epoch {epoch}...")
            attribute_test = "race" if attribute == 'race/skin_tone' else attribute
            curr_eval = eval_train.run_eval_train_return(model, model.preprocess, attribute=attribute_test, dataset=dataset_name)

            if best_details["eval"] < curr_eval: # only save if best
                torch.save(model.state_dict(), os.path.join(exp_dir, f"best.pth"))
                best_details["eval"] = curr_eval
                best_details["epoch"] = epoch 
                print("Checkpoint saved.")
                count = 0
            else:
                count += 1

            best_epoch, best_ABLE = best_details["epoch"], best_details["eval"]
            print(f"Evaluation completed. Best epoch: {best_epoch}; Best ABLE: {best_ABLE}; Early Stopping Count: {count}")
            
            sys.stdout.flush()  
            sys.stderr.flush()

    # once finish, conduct eval
    best_epoch = best_details["epoch"]
    print(f"Conducting evaluation for best epoch, Epoch {best_epoch}...")
    model = DebiasedCLIP(backbone, device=device, mlp1_hidden_size=mlp1_hidden_size, mlp2_hidden_size=mlp2_hidden_size, alpha=alpha).to(device)
    preprocess = model.preprocess
    model.load_state_dict(torch.load(os.path.join(exp_dir, f"best.pth")))
    results = run_eval(model, preprocess, attribute=attribute)
    print(results)
    sys.stdout.flush()  
    sys.stderr.flush()

if __name__ == '__main__':
    main()
