import json
import pprint
from multiprocessing.spawn import import_main_path
import torch.backends.cudnn as cudnn
from tqdm import tqdm, trange
import time
from PIL import Image
import os
import torchvision.transforms.functional as F
# import wandb
# from lavis.models import load_model
# wandb.init(project='timeblip_new')
import numpy as np
import matplotlib.pyplot as plt

from main.dataset import DatasetMR, start_end_collate_mr, prepare_batch_inputs_mr
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from main.config import BaseOptions, setup_model
from utils.basic_utils import set_seed, AverageMeter, dict_to_markdown
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM, fixscore, output_head_temporal, timeblip, SetCriterion
from base_models.matcher import build_matcher
from main.inference_mr import eval_epoch
from main.config import WarmupStepLR
from utils.model_utils import count_parameters

import torch
import torch.nn as nn
from collections import defaultdict
from lavis.models import load_model_and_preprocess
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    dataset_config = dict(
        image_root=opt.train_image_root,
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        v_feat_dim=opt.v_feat_dim,
        q_feat_dim=opt.t_feat_dim,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        use_cache=opt.use_cache,
        add_easy_negative=opt.add_easy_negative,
        easy_negative_only=opt.easy_negative_only
    )

    dataset_config["data_path"] = opt.train_path
    dataset_config["image_root"] = opt.train_image_root
    train_dataset = DatasetMR(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["image_root"] = opt.eval_image_root
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("txt_clip_asr", "txt_clip").replace("txt_clip_cap", "txt_clip")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        eval_dataset = DatasetMR(**dataset_config)
    else:
        eval_dataset = None

    blip2itm = Blip2ITM().to(device)
    '''加载原始预训练的模型权重'''
    checkpoint = torch.load('/mnt/bn/experience0313/gengwenjia/timeblip/checkpoints/blip2_pretrained.pth', map_location='cpu')
    incompatible_keys = blip2itm.load_state_dict(checkpoint['model'], strict=False)
    '''加载stage 1阶段预训练的权重'''
    # checkpoint = torch.load('/mnt/bn/experience0313/gengwenjia/timeblip/results/stage1_checkponits/checkpoint_epoch_45.pth', map_location='cpu')
    # incompatible_keys = blip2itm.load_state_dict(checkpoint['model_state_dict'], strict=False)
    pred_head = output_head_temporal(hidden_dim = 768, span_pred_dim = 2, num_frames= 75, batch_size = 4).to(device)
    model = timeblip(blip2itm, pred_head).to(device)

    # blip2itm = load_model("blip2_image_text_matching", "pretrain").to(device)     # load pretrained model
    # import ipdb; ipdb.set_trace()
    # pred_head = output_head_temporal(hidden_dim = 768, span_pred_dim = 2, num_frames= 75, batch_size = 4).to(device)
    # model = timeblip(blip2itm, pred_head).to(device)

    # for param in blip2itm.parameters():
    #     param.requires_grad = False

    '''之后进行criterion的构建'''
    matcher = build_matcher(opt)
    weight_dict = {"loss_b": opt.b_loss_coef,
                   "loss_g": opt.g_loss_coef,
                   "loss_f": opt.f_loss_coef,
                   "loss_h": opt.h_loss_coef,
                   "loss_s_intra": opt.s_loss_intra_coef,
                   "loss_s_inter": opt.s_loss_inter_coef}

    '''在这里首先不使用saliencyloss，因为这里涉及到了相似度的计算'''
    losses = ['spans', 'labels']

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict, losses=losses,
        eos_coef=opt.eos_coef, temperature=opt.temperature,
        span_loss_type=opt.span_loss_type, max_v_l=opt.max_v_l,
        saliency_margin=opt.saliency_margin,
    )
    criterion.to(device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    if opt.lr_warmup > 0:
        # total_steps = opt.n_epoch * len(train_dataset) // opt.bsz
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    if opt.lr_warmup != -1 and opt.lr_drop > 0:
        lr_scheduler = WarmupStepLR(optimizer, warmup_steps=opt.lr_warmup[0], step_size=opt.lr_drop, gamma=opt.lr_gamma)
    elif opt.lr_warmup != -1:
        from transformers import get_constant_schedule_with_warmup
        lr_scheduler =  get_constant_schedule_with_warmup(optimizer, opt.lr_warmup[0])

    elif opt.lr_drop > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=opt.lr_gamma)
    
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)

    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs_mr(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()

        # try:
        #import ipdb; ipdb.set_trace()
        outputs = model(model_inputs)
        '''在这里重新修正一下预测出来的片段的score'''
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        # wandb.log({"total_loss": losses, "loss_b": loss_dict['loss_b'], "loss_g": loss_dict['loss_g'], "loss_f": loss_dict['loss_f'], "loss_h": loss_dict['loss_h'], "epoch": epoch_i+1})
        #wandb.log({"total_loss": losses, "epoch": epoch_i+1})

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate_mr,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
        drop_last=True
    )

    prev_best_score = 0.
    es_cnt = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_init else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = opt.eval_epoch
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            # pdb.set_trace()
            stop_score = metrics["brief"][opt.main_metric]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if (epoch_i + 1) % opt.save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()
    # '''测试image-text matching所需要的输入'''


    # '''首先测试一下dataloader的正确性'''
    # for batch_idx, batch in enumerate(train_loader):
    #     model_inputs, targets = prepare_batch_inputs_mr(batch[1], opt.device, non_blocking=opt.pin_memory)
    #     #import ipdb; ipdb.set_trace()
    #     outputs = model(model_inputs)
    #     import ipdb; ipdb.set_trace()
    #     loss_dict = criterion(outputs, targets)
    #     import ipdb; ipdb.set_trace()
    #     weight_dict = criterion.weight_dict
    #     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
    #     import ipdb; ipdb.set_trace()


        # model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
        # orginal_score = []
        # index = 0
        # for i, item in enumerate(model_inputs['image_path']):
        #     current_path = item
        #     for image in os.listdir(current_path):
        #         current_image = Image.open(os.path.join(current_path, image))
        #         print('index is:', index)
        #         current_text = model_inputs['text_input'][index]
        #         img = vis_processors["eval"](current_image).unsqueeze(0).to(device)
        #         txt = text_processors["eval"](current_text)

        #         itm_output_ori = model({"image": img, "text_input": txt}, match_head="itm")
        #         itm_scores_ori = torch.nn.functional.softmax(itm_output_ori, dim=1)[:, 1].item()
        #         orginal_score.append(itm_scores_ori)
        #         index += 1
        #         print(orginal_score)
                # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    '''首先测试初步的image-textmatching'''
    best_ckpt_path, eval_split_name, eval_path, debug = start_training()