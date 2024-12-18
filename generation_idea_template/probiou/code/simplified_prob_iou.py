"""
1
Implement a new function `sp_iou(obb1, obb2, eps=1e-7)` that calculates a simplified probabilistic IoU
This function will use only variances (a, b) from the `_get_covariance_matrix` function, ignoring the covariance term (c)
2
Implement a hybrid function `hybrid_iou(obb1, obb2, threshold, eps=1e-7)` that uses the simplified approach when the rotation angle is below a threshold and the full `probiou` calculation otherwise
3
Implement a function `adaptive_hybrid_iou(obb1, obb2, validation_set, eps=1e-7)` that learns the optimal threshold from a validation set
4
Modify the `__main__` section to include tests that compare the output of `sp_iou`, `hybrid_iou`, `adaptive_hybrid_iou`, `probiou`, and `ciou` for various `obb1` and `obb2` configurations, specifically varying the rotation angle in `obb2`
5
Measure the execution time of `sp_iou`, `hybrid_iou`, `adaptive_hybrid_iou`, `probiou`, and `ciou` for a large number of bounding box pairs
6
Analyze the trade-off between accuracy (compared to `probiou`) and efficiency, including an analysis of the error introduced by ignoring rotation as a function of the rotation angle and the impact of the threshold in `hybrid_iou` and `adaptive_hybrid_iou`
7
Calculate the correlation between `sp_iou`, `hybrid_iou`, `adaptive_hybrid_iou`, and `probiou` to assess their agreement in ranking bounding box pairs
8
Train a simple object detection model with each of the IoU metrics as the loss function and evaluate their performance on a test set
9
Discuss the application-specific considerations for choosing the appropriate IoU metric

"""

# Modified code
import math
import torch
import time


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Note:
        OBB format: [center_x, center_y, width, height, rotation_angle].
        If CIoU is True, returns CIoU instead of IoU.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def sp_iou(obb1, obb2, eps=1e-7):
    """
    Simplified probabilistic IoU ignoring the covariance term.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): Simplified IoU scores, shape (N,).
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, _ = _get_covariance_matrix(obb1)
    a2, b2, _ = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) + eps)
    ) * 0.25
    t3 = (
        ((a1 + a2) * (b1 + b2))
        / (4 * ((a1 * b1).clamp_(0) * (a2 * b2).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def hybrid_iou(obb1, obb2, threshold, eps=1e-7):
    """
    Hybrid IoU that uses the simplified approach for small rotations.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        threshold (float): Rotation angle threshold in radians for switching between simplifications.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): Hybrid IoU scores, shape (N,).
    """
    rotation_diff = torch.abs(obb1[..., 4] - obb2[..., 4])
    use_simplified = rotation_diff < threshold
    ious = torch.where(use_simplified, sp_iou(obb1, obb2, eps), probiou(obb1, obb2, eps=eps))
    return ious


def adaptive_hybrid_iou(obb1, obb2, validation_set, eps=1e-7):
    """
    Adaptive hybrid IoU that learns the optimal threshold from a validation set.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        validation_set (list of tuples): List of (obb1, obb2) pairs for validation.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): Adaptive hybrid IoU scores, shape (N,).
    """
    best_threshold = 0.0
    best_score = float('inf')
    for threshold in torch.arange(0, math.pi / 2, 0.1):
        total_error = 0
        for val_obb1, val_obb2 in validation_set:
            hybrid_iou_val = hybrid_iou(val_obb1, val_obb2, threshold)
            prob_iou_val = probiou(val_obb1, val_obb2)
            total_error += torch.sum((hybrid_iou_val - prob_iou_val).abs())
        if total_error < best_score:
            best_score = total_error
            best_threshold = threshold

    return hybrid_iou(obb1, obb2, best_threshold, eps)


if __name__ == "__main__":
    # Define test OBBs
    obb1 = torch.tensor([[0.0, 0.0, 2.0, 4.0, 0.0]])  # [x, y, w, h, r=0°]
    obb2 = torch.tensor([[5.0, 5.0, 6.0, 2.0, math.radians(30)]])  # [x, y, w, h, r=30°]
    obb3 = torch.tensor([[5.0, 5.0, 6.0, 2.0, math.radians(45)]])  # [x, y, w, h, r=45°]

    # Test IoU functions
    print("OBB1:", obb1)
    print("OBB2:", obb2)
    print("OBB3:", obb3)

    # Calculate and compare IoUs
    iou = probiou(obb1, obb2, CIoU=False)
    print("ProbIoU (OBB1, OBB2):", iou)

    ciou = probiou(obb1, obb2, CIoU=True)
    print("CIoU (OBB1, OBB2):", ciou)

    sp_iou_val = sp_iou(obb1, obb2)
    print("Simplified ProbIoU (OBB1, OBB2):", sp_iou_val)

    hybrid_iou_val = hybrid_iou(obb1, obb2, threshold=math.radians(20))
    print("Hybrid IoU (OBB1, OBB2):", hybrid_iou_val)

    # Example validation set
    validation_set = [(obb1, obb2), (obb1, obb3)]
    adaptive_hybrid_iou_val = adaptive_hybrid_iou(obb1, obb2, validation_set)
    print("Adaptive Hybrid IoU (OBB1, OBB2):", adaptive_hybrid_iou_val)

    # Measure execution time
    num_pairs = 10000
    obbs1 = torch.rand((num_pairs, 5))
    obbs2 = torch.rand((num_pairs, 5))

    start = time.time()
    for _ in range(num_pairs):
        sp_iou(obbs1, obbs2)
    print("Time taken by Simplified ProbIoU:", time.time() - start)

    start = time.time()
    for _ in range(num_pairs):
        hybrid_iou(obbs1, obbs2, threshold=math.radians(20))
    print("Time taken by Hybrid IoU:", time.time() - start)

    start = time.time()
    for _ in range(num_pairs):
        adaptive_hybrid_iou(obbs1, obbs2, validation_set)
    print("Time taken by Adaptive Hybrid IoU:", time.time() - start)

    start = time.time()
    for _ in range(num_pairs):
        probiou(obbs1, obbs2)
    print("Time taken by ProbIoU:", time.time() - start)

    start = time.time()
    for _ in range(num_pairs):
        probiou(obbs1, obbs2, CIoU=True)
    print("Time taken by CIoU:", time.time() - start)