import torch
from torchvision.ops import nms


def post_process(img_batch, classification, regression, anchors, regress_boxes, clip_boxes):

    transformed_anchors = regress_boxes(anchors, regression)
    # validation 진행할때는 img는 1장씩 들어오게됨.
    transformed_anchors = clip_boxes(transformed_anchors, img_batch)

    finalScores = torch.Tensor([])
    finalAnchorBoxesIndexes = torch.Tensor([]).long()
    finalAnchorBoxesCoordinates = torch.Tensor([])

    if torch.cuda.is_available():
        finalScores = finalScores.cuda()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

    for i in range(classification.shape[2]):
        scores = torch.squeeze(classification[:, :, i])
        scores_over_thresh = (scores > 0.05)
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just continue
            continue
        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(transformed_anchors)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])

        if torch.cuda.is_available():
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def post_process_tmp(img_batch, classification, regression, anchors, regress_boxes, clip_boxes, entropy=False):

    transformed_anchors = regress_boxes(anchors, regression)
    # validation 진행할때는 img는 1장씩 들어오게됨.
    transformed_anchors = clip_boxes(transformed_anchors, img_batch)

    scores = torch.max(classification, dim=2, keepdim=True)[0]

    scores_over_thresh = (scores > 0.05)[0, :, 0]

    if scores_over_thresh.sum() == 0:
        print('No boxes to NMS')
        # no boxes to NMS, just return
        return [torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(0, 4)]

    classification = classification[:, scores_over_thresh, :]
    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
    scores = scores[:, scores_over_thresh, :]
    anchors_nms_idx = nms(
        transformed_anchors[0, :, :], scores[0, :, 0], iou_threshold=0.5)

    if not entropy:
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        return nms_class, nms_class, transformed_anchors[0, anchors_nms_idx, :]

    return classification[0, anchors_nms_idx, :]