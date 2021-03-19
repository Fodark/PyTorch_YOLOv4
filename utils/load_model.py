import torch
from models.yolo import Model


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

    with model.eval():
        def predict(data):
            pred = model(data, nc=80)
            return pred

        return predict
