import torch
from .models.yolo import Model
from .utils.utils import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)


class YoloMish:
    def __init__(self, cfg="models/yolov4l-mish.yaml", weights="weights/yolov4l-mish.pt", conf_thresh=.3, iou_thresh=.65):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(cfg).to(self.device)
        ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
        exclude = ['anchor']  # exclude keys
        ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                         if k in self.model.state_dict() and not any(x in k for x in exclude)
                         and self.model.state_dict()[k].shape == v.shape}
        self.model.load_state_dict(ckpt['model'], strict=False)
        print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(self.model.state_dict()), weights))

    def new_detect(self, data):
        data = torch.tensor(data)
        data = data.unsqueeze(0)
        self.model.eval()
        nb, _, height, width = data.shape  # batch size, channels, height, width
        inf_out, train_out = self.model(data)
        output = non_max_suppression(inf_out, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, merge=False)

        bboxes = []
        confidences = []
        class_ids = []
        clip_coords(output[0], (height, width))
        for pred in output[0]:
            *bbox, conf, class_id = pred.tolist()
            bboxes.append([round(b, 3) for b in bbox])
            confidences.extend([round(conf, 5)])
            class_ids.extend([int(class_id)])

        return bboxes, confidences, class_ids
