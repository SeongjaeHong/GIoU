import torch


class BBox_util:
    def validate_bbox(self, bbox):
        """Validate bbox if its values form (xmin, ymin, xmax, ymax).
        Args:
            bbox: (tensor) bounding boxes, Shape: [N, 4]
        Return:
            (bool) True if values are correct, otherwise False.
        """
        if (bbox[:, 2] > bbox[:, 0]).all() and (bbox[:, 3] > bbox[:, 1]).all():
            return True
        else:
            return False

    def intersection(self, box_g, box_p):
        """Resize both tensors to [num_objects, num_prediction,2]
        [num_objects, 2] -> [num_objects, 1, 2] -> [num_objects, num_prediction, 2]
        [num_prediction, 2] -> [1,num_prediction, 2] -> [num_objects, num_prediction, 2]
        Then compute the area of enclosure of box_g and box_p.
        Args:
            box_g: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4] (xmin, ymin, xmax, ymax)
            box_p: (tensor) Predicted bounding boxes, Shape: [num_prediction, 4] (xmin, ymin, xmax, ymax)
        Return:
            (tensor) intersection area, Shape: [num_objects, num_prediction].
        """
        A = box_g.size(0)
        B = box_p.size(0)
        min_xy = torch.max(box_g[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_p[:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy = torch.min(box_g[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_p[:, 2:].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def enclosure(self, box_g, box_p):
        """Resize both tensors to [num_objects, num_prediction,2]
        [num_objects, 2] -> [num_objects, 1, 2] -> [num_objects, num_prediction, 2]
        [num_prediction, 2] -> [1,num_prediction, 2] -> [num_objects, num_prediction, 2]
        Then compute the area of enclosure of box_g and box_p.
        Args:
            box_g: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4] (xmin, ymin, xmax, ymax)
            box_p: (tensor) Predicted bounding boxes, Shape: [num_prediction, 4] (xmin, ymin, xmax, ymax)
        Return:
            (tensor) enclosure area, Shape: [num_objects, num_prediction].
        """
        A = box_g.size(0)
        B = box_p.size(0)
        min_xy = torch.min(box_g[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_p[:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy = torch.max(box_g[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_p[:, 2:].unsqueeze(0).expand(A, B, 2))
        enc = torch.clamp((max_xy - min_xy), min=0)
        return enc[:, :, 0] * enc[:, :, 1]

    def iou(self, box_g, box_p):
        """Compute the jaccard overlap of two sets of boxes.
        The jaccard overlap is the intersection over union of two boxes.
        It operates on ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_g: (tensor) Ground truth bounding boxes, Shape: [num_objects,4] (xmin, ymin, xmax, ymax)
            box_p: (tensor) Predicted bounding boxes, Shape: [num_prediction,4] (xmin, ymin, xmax, ymax)
        Return:
            jaccard overlap: (tensor) Shape: [box_g.size(0), box_p.size(0)]
        """
        assert self.validate_bbox(box_g) and self.validate_bbox(box_p), "Bounding boxes are not correct."

        inter = self.intersection(box_g, box_p)
        area_a = ((box_g[:, 2] - box_g[:, 0]) *
                  (box_g[:, 3] - box_g[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_p[:, 2] - box_p[:, 0]) *
                  (box_p[:, 3] - box_p[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union

    def giou(self, box_g, box_p):
        """Compute the generalized IoU(GIoU) of two sets of boxes.
            It operates on ground truth boxes and default boxes.

            Reference:
                Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019).
                Generalized intersection over union: A metric and a loss for bounding box regression.
                https://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf

            Args:
                box_g: (tensor) Ground truth bounding boxes, Shape: [num_objects,4] (xmin, ymin, xmax, ymax)
                box_p: (tensor) Predicted bounding boxes, Shape: [num_predictions,4] (xmin, ymin, xmax, ymax)
            Return:
                jaccard overlap: (tensor) Shape: [box_g.size(0), box_p.size(0)]
            """
        assert self.validate_bbox(box_g) and self.validate_bbox(box_p), "Bounding boxes are not correct."

        inter = self.intersection(box_g, box_p)
        area_a = ((box_g[:, 2] - box_g[:, 0]) *
                  (box_g[:, 3] - box_g[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_p[:, 2] - box_p[:, 0]) *
                  (box_p[:, 3] - box_p[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        enc_area = self.enclosure(box_g, box_p)
        iou = inter / union
        return iou - (enc_area - union) / enc_area


class GIoULoss:
    def __call__(self, box_g, box_p):
        return self.forward(box_g, box_p)

    def forward(self, box_g, box_p):
        """Compute loss of generalized IoU(GIoU)
        Args:
            box_g: (tensor) Ground truth bounding boxes, Shape: [num_objects,4] (xmin, ymin, xmax, ymax)
            box_p: (tensor) Predicted bounding boxes, Shape: [num_predictions,4] (xmin, ymin, xmax, ymax)
        Return:
            loss of giou: (tensor) Shape: [box_g.size(0), box_p.size(0)]
        """
        return 1 - BBox_util().giou(box_g, box_p)


if __name__ == '__main__':
    gt = torch.Tensor([[4, 0, 7, 4], [4, 0, 7, 3]])
    prior = torch.Tensor([[1, 1, 5, 4], [2, 2, 8, 5]])
    bbox_util = BBox_util()
    loss = GIoULoss()

    """
    [[0.1429, 0.2500],
     [0.1053, 0.1250]]
    """
    print('IoU :', bbox_util.iou(gt, prior))

    """
    [[ 0.0179,  0.0500],
     [-0.1031, -0.0750]]
    """
    print('GIoU :', bbox_util.giou(gt, prior))

    """
    [[0.8571, 0.7500],
     [0.8947, 0.8750]]
    """
    print('Loss IoU :', 1 - bbox_util.iou(gt, prior))

    """
    [[0.9821, 0.9500],
     [1.1031, 1.0750]]
    """
    print('Loss GIoU :', loss(gt, prior))
