# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging as logging_
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage, Average
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import *
from model import *
import pickle as pkl

from dataset import get_dataset, AVSDDataSet, collate_fn
import pdb
import copy
#import ipdb

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

logger = logging_.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def get_data_loaders_new(args, tokenizer):
    train_data = get_dataset(tokenizer, args.train_path, args.fea_path, n_history=args.max_history)
    valid_data = get_dataset(tokenizer, args.valid_path, args.fea_path, n_history=args.max_history)
    train_dataset = AVSDDataSet(train_data[0], tokenizer, (train_data[1], valid_data[1]), drop_rate=0, train=True)
    valid_dataset = AVSDDataSet(valid_data[0], tokenizer, (valid_data[1], train_data[1]), drop_rate=0, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed), collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, num_workers=4, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id, features=True))
    return train_loader, valid_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/HEAR_dataset/SNE_dataset/train_set4DSTC7-AVSD.json", help="Path of the trainset")
    parser.add_argument("--valid_path", type=str, default="./data/HEAR_dataset/SNE_dataset/val_set4DSTC7-AVSD.json", help="Path of the validset")
    parser.add_argument("--fea_path", type=str, default="data/", help="Path of the trainset")
    parser.add_argument("--model_checkpoint", type=str, default="t5-large", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=3, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=3, help="Batch size for validation")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop rate for caption")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=9, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--log_path", type=str, default="log/exp2", help="Log path")
    parser.add_argument("--LE", action='store_true', help="If true start Listening Enhancement")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging_.basicConfig(level=logging_.INFO if args.local_rank in [-1, 0] else logging_.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = T5TokenizerFast
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = T5ForConditionalGeneration_AVSD
    
    # T5 Pretrain
    model = model_class.from_pretrained(args.model_checkpoint)
    
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader = get_data_loaders_new(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        
        input_ids = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        input_mask = batch[2].to(args.device)
        i3d = batch[3].to(args.device)
        RLE_input = batch[4].to(args.device)
        RLE_labels = batch[5].to(args.device)
        AR_mask = batch[6].to(args.device)
        RLE_input_bound = batch[7].to(args.device)
        
        # copy for alternative training
        aud_input_ids = copy.deepcopy(input_ids)
        aud_input_bound_ids = copy.deepcopy(input_ids)

        input_embs = model.encoder.embed_tokens(input_ids)
        aud_input_embs = model.encoder.embed_tokens(aud_input_ids)
        aud_input_bound_embs = model.encoder.embed_tokens(aud_input_bound_ids)


        i3d_embs = model.video_ff(i3d)
        input_embs = torch.cat([i3d_embs, input_embs], dim=1)
        iteration_odd = engine.state.iteration%2

        if iteration_odd == 1:
            RLE_embs = model.video_ff(RLE_input)
            RLE_input_embs = torch.cat([RLE_embs, aud_input_embs], dim=1)

            RLE_bound_embs = model.video_ff(RLE_input_bound)
            RLE_input_bound_embs = torch.cat([RLE_bound_embs, aud_input_bound_embs], dim=1)

            target_loss = model(inputs_embeds=input_embs, labels=labels, attention_mask=input_mask, aud=RLE_labels, is_LE=True, inputs_aud_embeds=RLE_input_embs, AR_mask=AR_mask, epoch=engine.state.epoch)
            bound_loss = model(inputs_embeds=input_embs, labels=labels, attention_mask=input_mask, aud=RLE_labels, is_LE=True, inputs_aud_embeds=RLE_input_bound_embs, AR_mask=AR_mask, epoch=engine.state.epoch)
            loss = torch.clip(target_loss - bound_loss + 0.01, min=0)
        else:
            loss = model(inputs_embeds=input_embs, labels=labels, attention_mask=input_mask).loss

        loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return loss.item()
    trainer = Engine(update)
    #trainer = DeterministicEngine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():

            input_ids = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            input_mask = batch[2].to(args.device)
            i3d = batch[3].to(args.device)
            RLE_input = batch[4].to(args.device)
            RLE_labels = batch[5].to(args.device)
            AR_mask = batch[6].to(args.device)
            
            aud_input_ids = copy.deepcopy(input_ids)

            input_embs = model.encoder.embed_tokens(input_ids)
            aud_input_embs = model.encoder.embed_tokens(aud_input_ids)

            i3d_embs = model.video_ff(i3d)
            input_embs = torch.cat([i3d_embs, input_embs], dim=1)

            RLE_embs = model.video_ff(RLE_input)
            RLE_input_embs = torch.cat([RLE_embs, aud_input_embs], dim=1)

            lm_logits = model(inputs_embeds=input_embs, labels=labels, attention_mask=input_mask, is_val=True, aud=RLE_labels, is_LE=False, inputs_aud_embeds=RLE_input_embs, AR_mask=AR_mask, epoch=engine.state.epoch).logits
            lm_logits_flat = lm_logits.view(-1, lm_logits.size(-1))
            lm_labels_flat = labels.view(-1)
            return lm_logits_flat, lm_labels_flat
            
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    
    if args.LE:
        metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0], x[1]))}
        metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
        metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
    else:
        metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0], x[1]))}
        metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
        metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=os.path.join(args.log_path, "tb_log"))
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.log_path, 'checkpoint', save_interval=1, n_saved=8,require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, os.path.join(args.log_path, 'model_training_args.bin'))
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.log_path, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_path)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.log_path, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
