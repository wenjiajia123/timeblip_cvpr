"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from torch import nn
import math
from utils.span_utils import generalized_temporal_iou

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

@registry.register_model("blip2_image_text_matching")
class Blip2ITM(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    def forward(self, samples, match_head="itc"):
        image = samples["image"]
        caption = samples["text_input"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        text = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_txt_len,
            padding=True,
            return_tensors="pt",
        ).to(image.device)

        if match_head == "itm":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                image.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, :query_tokens.size(1), :]

            # 计算 logits
            itm_logit = self.itm_head(itm_embeddings)
            itm_logit = itm_logit.mean(dim=1)

            return itm_embeddings, itm_logit

        elif match_head == "itc":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            import ipdb; ipdb.set_trace()
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
            sim, _ = torch.max(sims, dim=1)

            return sim

class Video_modeling(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="clip_base",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
        self.timelinear = nn.Linear(num_query_token, 75)
        

    def forward(self, itm_embeddings):

        # videoQformer forward pass
        video_queries = self.query_tokens.expand(itm_embeddings.shape[0], -1, -1)
        video_atts = torch.ones(itm_embeddings.size()[:-1], dtype=torch.long).to(itm_embeddings.device)
        
        video_output = self.Qformer.bert(
            query_embeds=video_queries,
            encoder_hidden_states=itm_embeddings,
            encoder_attention_mask=video_atts,
            return_dict=True,
        )

        final_video = self.timelinear(video_output.last_hidden_state[:, :video_queries.size(1), :].permute(0, 2, 1)).permute(0, 2, 1)

        return final_video

        
def fixscore(outputs, targets):
    pred_spans = outputs["pred_spans"]
    start_spans = targets['timestamp']
    pred_spans = start_spans + pred_spans
    spans = torch.clamp(pred_spans, min=0, max=1)
    spans = spans*pred_spans.shape[1]
    old_score = outputs['pred_logits'].squeeze(-1)
    new_score = torch.zeros_like(old_score)
    '''这里可以采取不同的分数构建方式'''
    for i in range(spans.shape[0]):
        for j in range(spans.shape[1]):
            start_frame = spans[i, j, 0]
            end_frame = spans[i, j, 1]
            span_score = torch.mean(old_score[i, int(start_frame):int(end_frame)])
            new_score[i, j] = span_score
    outputs['pred_logits'] = new_score.unsqueeze(-1)

    return outputs


class output_head_v2(nn.Module):
    def __init__(self, hidden_dim, span_pred_dim, batch_size, span_loss_type = "l1"):
        super().__init__()
        self.span_head = nn.Linear(hidden_dim, span_pred_dim)
        self.class_head = nn.Linear(hidden_dim, 1)
        self.batch_size = batch_size
        self.span_loss_type = span_loss_type
    
    def forward(self, itm_embeddings):
        span_logit = self.span_head(itm_embeddings)
        class_logit = self.class_head(itm_embeddings).sigmoid()
        if self.span_loss_type == "l1":
            span_logit = span_logit.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).cuda()
            idx_mask = idx_mask.repeat(span_logit.shape[0], span_logit.shape[1], 1)
            span_logit = span_logit * idx_mask    

        span_logit = span_logit.mean(dim=1)
        class_logit = class_logit.mean(dim=1)

        num_frames = int(span_logit.shape[0]/self.batch_size)

        return class_logit.reshape(self.batch_size, num_frames, -1), span_logit.reshape(self.batch_size, num_frames, -1)

class output_head_v3(nn.Module):
    def __init__(self, batch_size = 4, span_loss_type = "l1"):
        super().__init__()
        # self.span_head = nn.Conv1d(input_channels = 1, output_channels = 2, kernel_size = 10, stride = 1, padding = 0)
        # self.class_head = nn.Conv1d(input_channels = 1, output_channels = 1, kernel_size = 5 , stride = 1, padding = 0)
        self.span_head = nn.Conv1d(1, 2, 10, 1, "same")
        self.class_head = nn.Conv1d(1, 1, 5, 1, "same")
        self.batch_size = batch_size
        self.span_loss_type = span_loss_type

    def forward(self, itm_logit):
        score = torch.nn.functional.softmax(itm_logit, dim=1)[:, 1]
        score = score.reshape(self.batch_size, -1)
        score = score.unsqueeze(1)
        span_logit = self.span_head(score).permute(0, 2, 1)

        class_logit = self.class_head(score).permute(0, 2, 1).sigmoid()
        if self.span_loss_type == "l1":
            span_logit = span_logit.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).cuda()
            idx_mask = idx_mask.repeat(span_logit.shape[0], span_logit.shape[1], 1)
            span_logit = span_logit * idx_mask 
        
        num_frames = int(span_logit.shape[0]/self.batch_size)

        return class_logit, span_logit

class Permute(nn.Module):
    
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.transpose(-1, -2)

class ConvPyramid(nn.Module):
    
    def __init__(self, dims, strides):
        super(ConvPyramid, self).__init__()

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = nn.ReLU(inplace=True)
            else:
                layers = nn.Sequential()
                conv_cls = nn.Conv1d if p > 0 else nn.ConvTranspose1d
                for _ in range(abs(p)):
                    layers.extend([
                        Permute(),
                        conv_cls(dims, dims, 2, stride=2),
                        Permute(),
                        nn.LayerNorm(dims),
                        nn.ReLU(inplace=True)
                    ])
            self.blocks.append(layers)

        self.strides = strides

    def forward(self, x, mask, return_mask=False):
        pymid, pymid_msk = [], []

        for s, blk in zip(self.strides, self.blocks):
            if x.size(1) < s:
                continue

            pymid.append(blk(x))

            if return_mask:
                if s > 1:
                    msk = F.max_pool1d(mask.float(), s, stride=s).long()
                elif s < 1:
                    msk = mask.repeat_interleave(int(1 / s), dim=1)
                else:
                    msk = mask
                pymid_msk.append(msk)

        return pymid, pymid_msk

class VideoFeatureModeling(nn.Module):
    def __init__(self, feature_dim=768, num_frames=75, num_layers=3, num_heads=8, dropout=0.1):
        super(VideoFeatureModeling, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim * 4, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, feature_dim]
        x = x.permute(1, 0, 2)  # Change to [num_frames, batch_size, feature_dim]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to [batch_size, num_frames, feature_dim]
        return x

class VideoFeatureModeling_v2(nn.Module):
    def __init__(self, feature_dim=768, num_frames=75, num_layers=3, num_heads=8, dropout=0.1):
        super(VideoFeatureModeling_v2, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        self.position_encoding = nn.Parameter(torch.zeros(1, num_frames, feature_dim))
        nn.init.xavier_uniform_(self.position_encoding)  # Initialize position encoding

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim * 4, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, feature_dim]
        x = x + self.position_encoding  # Add position encoding
        x = x.permute(1, 0, 2)  # Change to [num_frames, batch_size, feature_dim]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to [batch_size, num_frames, feature_dim]
        return x


class VideoFeatureModeling_v3(nn.Module):
    def __init__(self, feature_dim=768, num_frames=75, num_layers=3, num_heads=8, dropout=0.1):
        super(VideoFeatureModeling_v3, self).__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames

        # Transformer for original scale
        self.transformer_encoder_scale1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # Transformer for half scale
        self.transformer_encoder_scale2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # Transformer for quarter scale
        self.transformer_encoder_scale3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout
            ),
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, feature_dim]
        batch_size, num_frames, feature_dim = x.size()

        # ------- Scale 1: Original Resolution -------
        x_scale1 = x.permute(1, 0, 2)  # Shape: [num_frames, batch_size, feature_dim]
        x_scale1_out = self.transformer_encoder_scale1(x_scale1)  # Shape: [num_frames, batch_size, feature_dim]

        # ------- Scale 2: Half Resolution -------
        # Downsample by factor of 2
        x_scale2 = F.avg_pool1d(
            x.permute(0, 2, 1),  # Change to [batch_size, feature_dim, num_frames]
            kernel_size=2,
            stride=2,
            padding=0
        )  # Shape: [batch_size, feature_dim, num_frames//2]
        x_scale2 = x_scale2.permute(2, 0, 1)  # Change to [num_frames//2, batch_size, feature_dim]
        x_scale2_out = self.transformer_encoder_scale2(x_scale2)  # Shape: [num_frames//2, batch_size, feature_dim]

        # Upsample back to original resolution
        x_scale2_out = F.interpolate(
            x_scale2_out.permute(1, 2, 0),  # [batch_size, feature_dim, num_frames//2]
            size=num_frames,
            mode='linear',
            align_corners=False
        )  # Shape: [batch_size, feature_dim, num_frames]
        x_scale2_out = x_scale2_out.permute(2, 0, 1)  # Shape: [num_frames, batch_size, feature_dim]

        # ------- Scale 3: Quarter Resolution -------
        # Downsample by factor of 4
        x_scale3 = F.avg_pool1d(
            x.permute(0, 2, 1),  # Change to [batch_size, feature_dim, num_frames]
            kernel_size=4,
            stride=4,
            padding=0
        )  # Shape: [batch_size, feature_dim, num_frames//4]
        x_scale3 = x_scale3.permute(2, 0, 1)  # Change to [num_frames//4, batch_size, feature_dim]
        x_scale3_out = self.transformer_encoder_scale3(x_scale3)  # Shape: [num_frames//4, batch_size, feature_dim]

        # Upsample back to original resolution
        x_scale3_out = F.interpolate(
            x_scale3_out.permute(1, 2, 0),  # [batch_size, feature_dim, num_frames//4]
            size=num_frames,
            mode='linear',
            align_corners=False
        )  # Shape: [batch_size, feature_dim, num_frames]
        x_scale3_out = x_scale3_out.permute(2, 0, 1)  # Shape: [num_frames, batch_size, feature_dim]

        # ------- Combine Multi-Scale Outputs -------
        x_out = x_scale1_out + x_scale2_out + x_scale3_out  # Shape: [num_frames, batch_size, feature_dim]
        x_out = x_out.permute(1, 0, 2)  # Change back to [batch_size, num_frames, feature_dim]

        return x_out

class LearnableQueryAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(LearnableQueryAggregator, self).__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, feature_dim))  # Learnable query
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x):
        x = x.transpose(0, 1)
        query = self.query.expand(1, x.size(1), x.size(2)) # Expand query to batch size
        attn_output, _ = self.multihead_attn(query, x, x)
        representation = attn_output.squeeze(0)
        return representation

class QueryMultiHeadAttentionAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(QueryMultiHeadAttentionAggregator, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

    def forward(self, x):
        """
        输入：
            x: [batch_size, num_queries, feature_dim]
        输出：
            representation: [batch_size, feature_dim]
        """
        x = x.transpose(0, 1)  # 转换为 [num_queries, batch_size, feature_dim]
        # 使用零向量作为查询，尺寸为 [1, batch_size, feature_dim]
        query = torch.zeros(1, x.size(1), x.size(2), device=x.device)
        # 计算注意力
        attn_output, _ = self.multihead_attn(query, x, x)
        representation = attn_output.squeeze(0)  # [batch_size, feature_dim]
        return representation
    
class output_head_temporal(nn.Module):
    def __init__(self, hidden_dim, span_pred_dim, num_frames, batch_size, span_loss_type = "l1"):
        super().__init__()
        self.patch_score = LearnableQueryAggregator(feature_dim=hidden_dim, num_heads=8)
        self.temporal = VideoFeatureModeling_v3(feature_dim = hidden_dim, num_frames = num_frames)
        #self.temporal = Video_modeling()
        self.span_embed = Conv(hidden_dim, hidden_dim, span_pred_dim, 3, kernel_size=3)
        self.class_embed = Conv(hidden_dim, hidden_dim, 1, 3, kernel_size=3)
        self.batch_size = batch_size
        self.span_loss_type = span_loss_type
    
    def forward(self, itm_embeddings, itm_logits):
        num_frames = int(itm_embeddings.shape[0]/self.batch_size)

        #itm_embeddings = itm_embeddings.mean(dim = 1).reshape(self.batch_size, num_frames, -1)
        itm_embeddings = self.patch_score(itm_embeddings).reshape(self.batch_size, num_frames, -1)
        #生成mask, 对于qvhighlights直接生成为全部为1的mask
        tem_embeddings = self.temporal(itm_embeddings)
        outputs_class = self.class_embed(tem_embeddings).sigmoid()  
        outputs_coord = self.span_embed(tem_embeddings) 

        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).cuda()
            idx_mask = idx_mask.repeat(outputs_coord.shape[0], outputs_coord.shape[1], 1)
            outputs_coord = outputs_coord * idx_mask   
       
        saliency_score = torch.nn.functional.softmax(itm_logits, dim=1)[:, 1].reshape(self.batch_size, -1)
  
        return outputs_class, outputs_coord, saliency_score

class output_head(nn.Module):
    def __init__(self, hidden_dim, span_pred_dim, batch_size, span_loss_type = "l1"):
        super().__init__()
        self.span_embed = Conv(hidden_dim, hidden_dim, span_pred_dim, 3, kernel_size=3)
        self.class_embed = Conv(hidden_dim, hidden_dim, 1, 3, kernel_size=3)
        self.batch_size = batch_size
        self.span_loss_type = span_loss_type

    def forward(self, itm_embeddings, itm_logits):
        outputs_class = self.class_embed(itm_embeddings).sigmoid()  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(itm_embeddings)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
            idx_mask = torch.tensor((-1, 1)).unsqueeze(0).unsqueeze(0).cuda()
            idx_mask = idx_mask.repeat(outputs_coord.shape[0], outputs_coord.shape[1], 1)
            outputs_coord = outputs_coord * idx_mask

        outputs_class = outputs_class.mean(dim = 1)
        outputs_coord = outputs_coord.mean(dim=1)

        num_frames = int(outputs_class.shape[0]/self.batch_size)

        saliency_score = torch.nn.functional.softmax(itm_logits, dim=1)[:, 1].reshape(self.batch_size, -1)

        return outputs_class.reshape(self.batch_size, num_frames, -1), outputs_coord.reshape(self.batch_size, num_frames, -1), saliency_score

class output_strategy(nn.Module):
    def __init__(self, span_thersholds = [4,8,12], batch_size = 4):
        super().__init__()
        self.span_thersholds = span_thersholds
        self.batch_size = batch_size

    def forward(self, scores):
        scores = scores.reshape(self.batch_size, -1)
        all_spans = []
        for i in range(self.batch_size):
            score_threshold = torch.mean(scores, dim = -1)[i]
            video_scores = scores[i, :]
            '''进行操作, 变为150'''
            video_scores = video_scores.repeat_interleave(2)
            pos_id = []
            for j in range(video_scores.shape[0]):
                if video_scores[j] > score_threshold :
                    pos_id.append(j)
            
            pred_relevant_windows = []
            for span_threshold in self.span_thersholds:
                start_id = 0
                end_id = 0
                while start_id < len(pos_id)-1:
                    end_id = start_id
                    while pos_id[end_id+1] - pos_id[end_id] < span_threshold:
                        end_id += 1
                        if end_id > len(pos_id)-2:
                            break
                    
                    '''计算这段的得分'''
                    if pos_id[start_id] != pos_id[end_id]:
                        score = sum(video_scores[pos_id[start_id]:pos_id[end_id]+1]) / (pos_id[end_id] - pos_id[start_id] + 1)
                        if [pos_id[start_id]/150, pos_id[end_id]/150, score.item()] not in pred_relevant_windows:
                            pred_relevant_windows.append([pos_id[start_id]/150, pos_id[end_id]/150, score.item()])

                    
                    start_id = end_id+1
                    print('start_id:', start_id)
            all_spans += pred_relevant_windows

        sorted_spans = sorted(all_spans, key=lambda x: x[2], reverse=True)
        return sorted_spans


class Conv(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, groups=1, bias=True, padding_mode='zeros')
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        x = x.permute(0,2,1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.permute(0, 2, 1)
        
class timeblip(nn.Module):
    def __init__(self, blip2itm, pred_head):
        super().__init__()
        self.blip2itm = blip2itm
        self.pred_head = pred_head

    def forward(self, samples):
        itm_embeddings, itm_logit = self.blip2itm(samples, match_head="itm")
        outputs_class, outputs_coord, scores = self.pred_head(itm_embeddings, itm_logit)

        out = {'pred_logits': outputs_class, 'pred_spans': outputs_coord, 'saliency_scores': scores}
        
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.temperature = 0.07

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        assert 'pred_spans' in outputs

        start_spans = targets['timestamp']
        pred_spans = outputs['pred_spans']
        src_spans = start_spans + pred_spans
        gt_spans = targets['span_labels_nn']

        mask =  targets['timestamp_mask'].bool()
        mask_full = targets['timestamp_mask'].unsqueeze(2).repeat(1, 1, 2)
        mask_valid =  targets['timestamp_window'].bool()
        mask_valid_full = targets['timestamp_window'].unsqueeze(2).repeat(1, 1, 2)

        #loss_span = F.l1_loss(src_spans, gt_spans, reduction='none') * mask_valid_full
        loss_span = F.smooth_l1_loss(src_spans, gt_spans, reduction='none') * mask_valid_full
        original_loss_giou = 1 - torch.diag(generalized_temporal_iou(src_spans[mask_valid], gt_spans[mask_valid])[0])
        giou = torch.diag(generalized_temporal_iou(src_spans[mask_valid], gt_spans[mask_valid])[0])
        weight = torch.ones_like(giou)
        # 找出所有介于0.5和0.75之间的元素位置，并将对应位置设置为2
        mask2 = (giou > 0.5) & (giou <= 0.75)
        weight[mask2] = 2
        
        # 找出所有大于0.75的元素位置，并将对应位置设置为3
        mask3 = giou  > 0.75
        weight[mask3] = 3

        # 计算最终的Focal-IoU Loss
        loss_giou = weight * (1 - giou)
        loss_high = 1 - torch.diag(generalized_temporal_iou(src_spans[mask_valid], gt_spans[mask_valid])[1])

        losses = {}
        losses['loss_b'] = loss_span.sum() / mask_valid.sum()
        losses['loss_g'] = loss_giou.mean()
        #losses['loss_h'] = loss_high.mean()
        return losses

    def loss_strategy(self, outputs, targets, indices):
        gt = targets['span_labels']
        return -1
        
    def loss_labels(self, outputs, targets, indices, log=True):
        src_logits = outputs['pred_logits'].squeeze(-1)  # (batch_size, #queries, #classes=2)
        mask = targets['timestamp_mask'].bool()
        mask_valid = targets['timestamp_window'].bool()
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[mask_valid] = 1
        # target_classes = targets['timestamp_window']  # soft cls.
        target_classes.float()
        # pdb.set_trace()

        weights = torch.zeros_like(target_classes).float()
        weights[mask] = self.empty_weight[1]
        weights[mask_valid] = self.empty_weight[0]

        # pdb.set_trace()
        loss_ce = F.binary_cross_entropy(src_logits, target_classes.float(), weight=weights,  reduction="none") * mask
        loss_highlight = F.binary_cross_entropy(outputs['saliency_scores'], target_classes.float(), weight=weights,  reduction="none") * mask
        return {"loss_f": loss_ce.sum() / mask.sum(), "loss_h": loss_highlight.sum() / mask.sum()}
        # return {"loss_f": loss_ce.sum() / (1 + mask_valid.sum())}

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        # * inter-vid mode
        vid_mem_proj = outputs["vid_mem_proj"]
        pos_indices = targets["saliency_pos_labels"][:,0].long()  # (N, #pairs)
        batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        vid_feats = vid_mem_proj[batch_indices, pos_indices]
        txt_feats = outputs["txt_mem_proj"].squeeze(1)
        sim = sim_matrix(vid_feats, txt_feats)

        i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        jdiag = torch.diag(j_logsm)
        loss_i = idiag.sum() / len(idiag)
        loss_j = jdiag.sum() / len(jdiag)

        loss_saliency_inter = - loss_i - loss_j

        # * intra-vid mode
        mask = targets['timestamp_mask']
        selected_scores = saliency_scores[batch_indices, pos_indices].unsqueeze(-1)
        neg_indices_in = (saliency_scores < selected_scores)
        neg_indices_in[batch_indices, pos_indices] = True
        mask_invalid = neg_indices_in * mask.bool()

        sim_in = F.cosine_similarity(vid_mem_proj, txt_feats.unsqueeze(1), dim=-1)
        sim_in = sim_in + (mask_invalid + 1e-45).log()
        logsm_in_i = F.log_softmax(sim_in / self.temperature, dim=1)
        logsm_in_j = F.log_softmax(sim_in.t() / self.temperature, dim=1)

        pos_logsm_in_i = logsm_in_i[batch_indices, pos_indices]
        pos_logsm_in_j = logsm_in_j[pos_indices, batch_indices]
        loss_in_i = pos_logsm_in_i.sum() / len(pos_logsm_in_i)
        loss_in_j = pos_logsm_in_j.sum() / len(pos_logsm_in_j)

        loss_saliency_intra = - loss_in_i - loss_in_j

        return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def loss_saliency_cls(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}
        saliency_scores = targets["saliency_scores"]
        if saliency_scores.sum() == 0:
            return {"loss_s_inter": 0., "loss_s_intra": 0.}

        # * inter-vid mode
        vid_mem_proj = outputs["vid_mem_proj"]
        pos_indices = targets["saliency_pos_labels"][:,0].long()  # (N, #pairs)
        batch_indices = torch.arange(len(vid_mem_proj)).to(vid_mem_proj.device)

        vid_feats = vid_mem_proj[batch_indices, pos_indices]
        txt_feats = outputs["txt_mem_proj"].squeeze(1)
        sim = sim_matrix(vid_feats, txt_feats)

        i_logsm = F.log_softmax(sim / self.temperature, dim=1)
        j_logsm = F.log_softmax(sim.t() /self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        jdiag = torch.diag(j_logsm)
        loss_i = idiag.sum() / len(idiag)
        loss_j = jdiag.sum() / len(jdiag)

        loss_saliency_inter = - loss_i - loss_j

        # * intra-vid mode
        if 'cls_idx' not in targets.keys(): # eval
            return {"loss_s_inter": loss_saliency_inter}

        cls_indices = targets['cls_idx'].bool()
        cls_feats = outputs["cls_mem_proj"].squeeze(1)
        sim_cls = sim_matrix(vid_feats, cls_feats)

        i_logsm_cls = F.log_softmax(sim_cls / self.temperature, dim=1)
        idiag_cls = i_logsm_cls[cls_indices]
        loss_cls_i = idiag_cls.sum() / len(idiag_cls)

        loss_saliency_intra = - loss_cls_i

        return {"loss_s_inter": loss_saliency_inter, "loss_s_intra": loss_saliency_intra}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "saliency_cls": self.loss_saliency_cls,
            "strategy": self.loss_strategy,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, hl_only=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = None
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        return losses
