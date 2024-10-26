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

gt = [[1, 2], [5, 6]]
pred = [5, 6]
iou_threshold = 0.3

recall = calculate_recall(gt, pred, iou_threshold)
print(f"Recall: {recall}")