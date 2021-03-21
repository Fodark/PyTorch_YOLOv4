import torch
from models.yolo import Model
from utils.utils import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)


def load_model(cfg="models/yolov4l-mish.yaml", weights="weights/yolov4l-mish.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    exclude = ['anchor']  # exclude keys
    ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                     if k in model.state_dict() and not any(x in k for x in exclude)
                     and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(ckpt['model'], strict=False)
    print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))

    def predict(data, conf_thresh=.3, iou_thresh=.65, merge=False):
        model.eval()
        jdict = []
        nb, _, height, width = data.shape  # batch size, channels, height, width
        inf_out, train_out = model(data)
        output = non_max_suppression(inf_out, conf_thres=conf_thresh, iou_thres=iou_thresh, merge=merge)

        for si, pred in enumerate(output):
            clip_coords(pred, (height, width))
            box = pred[:, :4].clone()  # xyxy
            #scale_coords([3, height, width], box, height, width)  # to original shape
            box = xyxy2xywh(box)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                jdict.append({'class_id': int(p[5]),
                              'bbox': [round(x, 3) for x in b],
                              'score': round(p[4], 5)})

        return jdict

    return predict
