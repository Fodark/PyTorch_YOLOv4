import torch
from ..models.yolo import Model
from torchvision import transforms
from .utils import (
    non_max_suppression, clip_coords)


class YoloMish:
    def __init__(self, cfg="models/yolov4l-mish.yaml", weights="weights/yolov4l-mish.pt", conf_thresh=.3, iou_thresh=.65):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

        self.model = Model(cfg).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        print(f'Loaded {weights}')

    def new_detect(self, data):
        width, height = data.size 
        data = self.input_transform(data).to(self.device)
        data = data.unsqueeze(0)
        self.model.eval()
        inf_out, train_out = self.model(data)
        # run NMS
        output = non_max_suppression(inf_out, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, merge=False)
        output = output[0]

        boxes = output[:, :4].clone()
        

        # scale back from 640x640 to original size
        boxes[:, [0,2]] *= width / 640
        boxes[:, [1,3]] *= height / 640
        # go from xyxy to tlwh
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # clip coords to be valid
        clip_coords(boxes, (height, width))

        bboxes = boxes.tolist()
        confidences = []
        class_ids = []
        
        for pred in output:
            *tmp, conf, class_id = pred.tolist()
            confidences.extend([round(conf, 5)])
            class_ids.extend([int(class_id)])

        return bboxes, confidences, class_ids
