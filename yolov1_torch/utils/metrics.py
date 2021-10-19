import torch

class IoU(torch.nn.Module): 
    
    def __init__(self,):
        super(IoU,self).__init__()
    
    def forward(self,box1,box2,xywh=True,eps=1e-7): 
        """compute Iou between box1 (ground-truth-box) and box2 (predict-box)

        Args:
            box1 (torch.Tensor): x,y,w,h in [0,1] range
            box2 (torch.Tensor): the same as box1
        """
        if xywh: 
            # convert xywh to xyxy
            b1_x1 = box1[0] - box1[2] / 2
            b1_y1 = box1[1] - box1[3] / 2
            b1_x2 = box1[0] + box1[2] / 2
            b1_y2 = box1[1] + box1[3] / 2

            b2_x1 = box2[0] - box2[2] / 2
            b2_y1 = box2[1] - box2[3] / 2
            b2_x2 = box2[0] + box2[2] / 2
            b2_y2 = box2[1] + box2[3] / 2
        else:
            b1_x1,b1_y1,b1_x2,b1_y2 = box1
            b2_x1,b2_y1,b2_x2,b2_y2 = box2

        # Compute Intersection area
        intersection = (torch.min(b1_x2,b2_x2) -  torch.max(b1_x1,b2_x1)).clamp(0) * \
                       (torch.min(b1_y2,b2_y2) - torch.max(b1_y1,b2_y1)).clamp(0) 
        
        # Compute area for sigle box
        area_1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1 + eps)
        area_2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1 + eps)
        union = area_1 + area_2 - intersection + eps

        iou = union / intersection

        return iou

