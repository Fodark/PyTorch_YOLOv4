import torch
from ..models.yolo import Model
from torchvision import transforms
from .utils import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)


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
        #ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
        #exclude = ['anchor']  # exclude keys
        #ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
        #                 if k in self.model.state_dict() and not any(x in k for x in exclude)
        #                 and self.model.state_dict()[k].shape == v.shape}
        self.model.load_state_dict(torch.load(weights))
        print(f'Loaded {weights}')

    def new_detect(self, data):
        width, height = data.size 
        data = self.input_transform(data).to(self.device)
        data = data.unsqueeze(0)
        self.model.eval()
        inf_out, train_out = self.model(data)
        output = non_max_suppression(inf_out, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, merge=False)
        output = output[0]

        boxes = output[:, :4].clone()
        print('INF')
        print(boxes)
        boxes[:, [0,2]] *= width / 640
        boxes[:, [1,3]] *= height / 640
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        print(boxes)

        #scale_coords((640, 640), boxes, (height, width))  # to original
        #clip_coords(output, (height, width))
        #boxes = xyxy2xywh(boxes)  # xywh
        #boxes[:, :2] -= boxes[:, 2:] / 2  # xy center to top-left corner

        bboxes = []
        confidences = []
        class_ids = []
        ##clip_coords(output[0], (height, width))
        for pred in output:
            *bbox, conf, class_id = pred.tolist()
            bboxes.append([round(b, 3) for b in bbox])
            confidences.extend([round(conf, 5)])
            class_ids.extend([int(class_id)])
        
        #bboxes = scale_coords((640, 640), bboxes, (height, width))
        #clip_coords(bboxes, (height, width))

        return bboxes, confidences, class_ids
