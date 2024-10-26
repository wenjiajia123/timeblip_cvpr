import torch
from tqdm import tqdm
from main.dataset import DatasetMR, start_end_collate_mr, prepare_batch_inputs_mr
from main.config import BaseOptions
from torch.utils.data import DataLoader
from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

opt = BaseOptions().parse()

save_dir = '/mnt/bn/duanbaishan-data/gengwenjia/timeblip/scores'
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

if opt.eval_path is not None:
    dataset_config["image_root"] = opt.eval_image_root
    dataset_config["data_path"] = opt.eval_path
    dataset_config["txt_drop_ratio"] = 0
    dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("txt_clip_asr", "txt_clip").replace("txt_clip_cap", "txt_clip")  # for pretraining
    # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
    eval_dataset = DatasetMR(**dataset_config)

eval_loader = DataLoader(
    eval_dataset,
    collate_fn=start_end_collate_mr,
    batch_size=opt.eval_bsz,
    num_workers=opt.num_workers,
    shuffle=False,
    pin_memory=opt.pin_memory,
    drop_last=True
)

blip2itm = Blip2ITM().to(device)
'''加载原始预训练的模型权重'''
checkpoint = torch.load('/mnt/bn/duanbaishan-data/gengwenjia/timeblip/checkpoints/blip2_pretrained.pth', map_location='cpu')
blip2itm.load_state_dict(checkpoint['model'], strict=False)
# pred_head = output_head(hidden_dim = 768, span_pred_dim = 2, batch_size = 4).to(device)
# model = timeblip(blip2itm, pred_head).to(device)
model = blip2itm.to(device)

for batch_idx, batch in tqdm(enumerate(eval_loader)):

    model_inputs, targets = prepare_batch_inputs_mr(batch[1], opt.device, non_blocking=opt.pin_memory)
    outputs = model(model_inputs)
    outputs1 = model(model_inputs, match_head = 'itc').squeeze(-1).cpu().detach().numpy()
    scores = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()

    # 创建x轴坐标
    x = np.arange(300)

    # 创建图形对象
    fig, ax = plt.subplots()

    # 绘制两个一维张量
    ax.plot(x, outputs1, label='itc scores')
    ax.plot(x, scores, label='itm scores')

    # 绘制gt
    sapn_gt = []
    for batch_id in range(4):
        spans_ori = targets['span_labels_nn'][batch_id,:,:]
        for i in range(spans_ori.shape[0]):
            if spans_ori[i,0] != 0 and spans_ori[i,1] != 0:
                sapn_gt.append([spans_ori[i,0].item()*75 + batch_id*75, spans_ori[i,1].item()*75 + batch_id*75])
        
    unique_data = list(set(map(tuple, sapn_gt)))

    # 将唯一的元素转换回列表
    gts = [list(item) for item in unique_data]

    
    for gt in gts:
        ax.plot(gt, [1.2, 1.2], linestyle='--', color='r', label='y=1')


    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 保存图像
    save_path = os.path.join(save_dir, f'tensor_plot_batch_{batch_idx}.png')
    plt.savefig(save_path)

    plt.close(fig)

