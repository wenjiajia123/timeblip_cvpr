def compute_iou(interval1, interval2):
    a, b = sorted(interval1)
    c, d = sorted(interval2)

    # 计算交集
    intersection = max(0, min(b, d) - max(a, c))
    
    # 计算并集
    union = (b - a) + (d - c) - intersection
    
    # 计算IoU
    iou = intersection / union if union != 0 else 0
    return iou

def smooth_l1_loss(x, y):
    diff = abs(x - y)
    if diff < 1.0:
        return 0.5 * diff**2
    else:
        return diff - 0.5

def calculate_giou(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    
    # Ensure intervals are valid
    start1, end1 = sorted([start1, end1])
    start2, end2 = sorted([start2, end2])
    
    # Calculate intersection
    intersection = max(0, min(end1, end2) - max(start1, start2))
    
    # Calculate closure
    closure = max(end1, end2) - min(start1, start2)
    
    # Calculate union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    # Calculate GIoU
    if closure == 0:  # avoid division by zero
        giou = -1
    else:
        giou = (intersection / union) - ((closure - union) / closure)
    
    return giou


# 定义区间
interval1 = [105/150, 122/150]
interval2 = [88.7161/150, 123.3556/150]

# 计算IoU
iou = compute_iou(interval1, interval2)
print("IoU:", iou)

# Calculate GIoU
giou_loss = calculate_giou(interval1, interval2)
print("GIoU Loss:", giou_loss)

# 计算每个元素的Smooth L1 Loss
losses = [smooth_l1_loss(interval1[i], interval2[i]) for i in range(2)]
print("Smooth L1 Losses:", losses)