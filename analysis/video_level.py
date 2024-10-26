import jsonlines
import json
import csv

def calculate_recall(gt, pred, iou_threshold):
    true_positives = 0
    total_gt = len(gt)
    
    for gt_interval in gt:
        iou = calculate_iou(gt_interval, pred)
        if iou >= iou_threshold:
            true_positives += 1
    
    recall = true_positives / total_gt
    return recall

def calculate_iou(interval1, interval2):
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    union = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])
    iou = intersection / union
    return iou

  
def read_jsonlines(file_path):  
    with open(file_path, 'r') as file:  
        lines = file.readlines()  
        data = [json.loads(line) for line in lines]  
    return data  

result_dir_list = [
    #'/mnt/bn/experience0313/gengwenjia/UVCOM/results/qvhighlights_results/hl-video_tef-test/best_hl_val_preds_nms_thd_0.7.jsonl',
    '/mnt/bn/experience0313/gengwenjia/timeblip/results/mr-qvhighlights/base1024_10_f10_b10g1_s0.05_0.01-slowfast_clip-clip-2024_06_11_19/best_qvhighlights_val_preds_nms_thd_0.7.jsonl',
    '/mnt/bn/experience0313/gengwenjia/timeblip/results/mr-qvhighlights/base1024_10_f10_b10g1_s0.05_0.01-slowfast_clip-clip-2024_07_08_17/best_qvhighlights_val_preds_nms_thd_0.7.jsonl'
]
#result_dir = '/mnt/bn/motor-cv-yzh/timeblip/results/mr-qvhighlights/base1024_10_f10_b10g1_s0.05_0.01-slowfast_clip-clip-2024_04_11_22/best_qvhighlights_val_preds_nms_thd_0.7.jsonl'
eval_dir = '/mnt/bn/duanbaishan-data/gengwenjia/dataset/qvhighlights/metadata/qvhighlights_val.jsonl'
save_file = '/mnt/bn/experience0313/gengwenjia/timeblip/analysis/video.csv'
threshold = 0.5

eval_anno = read_jsonlines(eval_dir)

def write_lists_to_csv(data_lists, file_path):
    data_rows = []
    for i in range(len(data_lists[0])):
        row = [data[i] for data in data_lists]
        data_rows.append(row)
        
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_rows)

all_infor = []
for result_dir in result_dir_list:
    score_th3 = []
    score_th5 = []
    score_th7 = []
    video_id = []
    gt = []
    pred = []
    result = read_jsonlines(result_dir)
    cnt03 = 0
    cnt05 = 0
    cnt07 = 0
    for res in result:
        qid = res['qid']
        for ann in eval_anno:
            if ann['qid'] == qid:
                relevant_clip_spans = ann['relevant_windows']
                r1_03 = calculate_recall(relevant_clip_spans, res['pred_relevant_windows'][0][:2], 0.3)
                r1_05 = calculate_recall(relevant_clip_spans, res['pred_relevant_windows'][0][:2], 0.5)
                r1_07 = calculate_recall(relevant_clip_spans, res['pred_relevant_windows'][0][:2], 0.7)
                score_th3.append(r1_03)
                score_th5.append(r1_05)
                score_th7.append(r1_07)
                video_id.append(qid)
                gt.append(relevant_clip_spans)
                pred.append(res['pred_relevant_windows'][0][:2])
  
    all_infor.append(video_id)
    all_infor.append(score_th3)
    all_infor.append(score_th5)
    all_infor.append(score_th7)
    all_infor.append(gt)
    all_infor.append(pred)
    
write_lists_to_csv(all_infor, save_file)

    