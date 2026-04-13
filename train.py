import os
import sys
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from model.attn_encoder import AttnEncoderXL
from utils.data_utils import ReactionDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from settings import Args
from model.flow_matching import ConditionalFlowMatcher
from utils.train_utils import get_lr, grad_norm, log_rank_0, NoamLR, \
    param_count, param_norm, set_seed, setup_logger, log_args
from torch.nn.init import xavier_uniform_
import torch.optim as optim

torch.set_printoptions(precision=4, profile="full", sci_mode=False, linewidth=10000)
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True, linewidth=500)

def init_dist(args):
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend,
                                init_method='env://',
                                timeout=datetime.timedelta(minutes=10))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    if dist.is_initialized():
        logging.info(f"Device rank: {dist.get_rank()}")
        sys.stdout.flush()


def init_model(args):
    state = {}
    if args.load_from:
        log_rank_0(f"Loading pretrained state from {args.load_from}")
        state = torch.load(args.load_from, map_location=torch.device("cpu"))
        pretrain_args = state["args"]
        pretrain_args.local_rank = args.local_rank

        graph_attn_model = AttnEncoderXL(pretrain_args)
        pretrain_state_dict = state["state_dict"]
        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        graph_attn_model.load_state_dict(pretrain_state_dict)
        log_rank_0("Loaded pretrained model state_dict.")
        flow_model = ConditionalFlowMatcher(args)
    else:
        graph_attn_model = AttnEncoderXL(args)
        flow_model = ConditionalFlowMatcher(args)
        for p in graph_attn_model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    graph_attn_model.to(args.device)
    flow_model.to(args.device)
    if args.local_rank != -1:
        graph_attn_model = DDP(
            graph_attn_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        log_rank_0("DDP setup finished")

    os.makedirs(args.model_path, exist_ok=True)

    return graph_attn_model, flow_model, state

def init_loader(args, dataset, batch_size: int, bucket_size: int = 1000,
                shuffle: bool = False, epoch: int = None, use_sort: bool =True):
    if use_sort: dataset.sort()
    if shuffle: dataset.shuffle_in_bucket(bucket_size=bucket_size)
    dataset.batch(
        batch_type=args.batch_type,
        batch_size=batch_size
    )

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        if epoch is not None:
            sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda _batch: _batch[0],
        pin_memory=True
    )

    return loader

def get_optimizer_and_scheduler(args, model, state=None):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    # scheduler = None
    scheduler = NoamLR(
        optimizer,
        model_size=args.emb_dim,
        warmup_steps=args.warmup_steps
    )
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=args.eval_iter, gamma=0.99
    # )

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler

def _optimize(args, model, optimizer, scheduler):
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    scheduler.step()
    g_norm = grad_norm(model)
    model.zero_grad(set_to_none=True)
    return g_norm

def main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    init_dist(args)
    log_args(args, 'training')
    model, flow, state = init_model(args)
    total_step = state["total_step"] if state else 0
    log_rank_0(f"Number of parameters: {param_count(model)}")

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, state)

    log_rank_0(f"Initializing training ...")
    log_rank_0(f"Loading data ...")
    with open(args.train_path, 'r') as train_o:
        train_smiles_list = train_o.readlines()
    with open(args.val_path, 'r') as val_o:
        val_smiles_list = val_o.readlines()
    
    train_dataset = ReactionDataset(args, train_smiles_list)
    val_dataset = ReactionDataset(args, val_smiles_list)

    accum = 0
    g_norm = 0
    losses, accs = [], []
    o_start = time.time()
    log_rank_0("Start training")

    accuracy = []
    for epoch in range(args.epoch):
        log_rank_0(f"Epoch: {epoch}")
        train_loader = init_loader(args, train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                epoch=epoch)
        model.train()
        model.zero_grad(set_to_none=True)
        for train_batch in train_loader:
            if total_step > args.max_steps:
                log_rank_0("Max steps reached, finish training")
                exit(0)

            train_batch.to(device)
            

            y = train_batch.src_token_ids
            y_len = train_batch.src_lens
            x0 = train_batch.src_matrices
            x1 = train_batch.tgt_matrices
            matrix_masks = train_batch.matrix_masks
            

            x0_sample = flow.sample_be_matrix(x0)

            t = torch.rand(x0.shape[0]).type_as(x0)
            
            xt = flow.sample_conditional_pt(x0, x1, t)
            ut = flow.compute_conditional_vector_field(x0_sample, x1)

            vt = model(y, y_len, xt, t)

            loss = (vt - ut) * matrix_masks
            if args.weight_uniform:
                loss = torch.sum((loss) ** 2) / matrix_masks.sum()
            else:
                loss = torch.sum((loss) ** 2) / loss.shape[0]
            (loss / args.accumulation_count).backward()
            losses.append(loss.item())

            accum += 1
            if accum == args.accumulation_count:
                g_norm = _optimize(args, model, optimizer, scheduler)
                accum = 0
                total_step += 1

            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                log_rank_0(f"Step {total_step}, loss: {np.mean(losses): .4f}, "
                        #    f"acc: {np.mean(accs): .4f},
                           f"p_norm: {param_norm(model): .4f}, g_norm: {g_norm: .4f}, "
                           f"lr: {get_lr(optimizer): .6f}, "
                           f"elapsed time: {time.time() - o_start: .0f}")
                losses, acc = [], []

            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                val_count = 50
                val_loader = init_loader(args, val_dataset,
                                        batch_size=args.val_batch_size,
                                        shuffle=True,
                                        epoch=epoch)
                from eval_multiGPU import get_predictions
                metrics = get_predictions(args, model, flow, val_loader, val_count)
                if dist.get_rank() == 0:
                    metrics = np.array(metrics)
                    log_rank_0(metrics.shape)
                    topk_accuracies = np.mean(metrics[:, 0].astype(bool)) # correct smiles
                    log_rank_0(f"Topk accuracies: {(topk_accuracies * 100): .2f}")
                model.train()

            # Important: saving only at one node or the ckpt would be corrupted!
            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1
                log_rank_0(f"Saving at step {total_step}")
                if scheduler is not None:
                    state = {
                        "args": args,
                        "total_step": total_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }
                else:
                    state = {
                        "args": args,
                        "total_step": total_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                torch.save(state, os.path.join(args.model_path, f"model.{total_step}_{n_iter}.pt"))

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            _optimize(args, model, optimizer, scheduler)
            accum = 0
            # total_step += 1           # for partial batch, do not increase total_step

        if args.local_rank != -1:
            dist.barrier()
    log_rank_0("Epoch ended")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = Args
    logger = setup_logger(args, "train")
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    main(args)
