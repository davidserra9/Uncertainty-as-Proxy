from torch import nn
from utils.NN_functions import *
from utils.config_parser import load_yml
from utils.UW_dataset import UWDataset
import numpy as np
import torch.nn.functional as F
import sys
import albumentations as A

cfg = load_yml("config.yml")

model = initialize_model(model_name=cfg.species_classification.model,
                         num_classes=len(cfg.species),
                         load_model=True,
                         balance=cfg.species_classification.balance,
                         data_aug=cfg.species_classification.data_aug,
                         model_root=cfg.model_path)

model.to('cuda')
generate_CAMs(folder_path=join(cfg.species_dataset, f"split_{cfg.species_classification.test_splits[0]}"),
              model=model,
              list_classes=cfg.species,
              n_images=5,
              output_path="",
              device="cuda")


model.to('cpu')
transformations = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                std=[0.1263, 0.1265, 0.1169]),
    ToTensorV2()
])

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.05 + orig_image * 1
        # put class label text on the result
        cv2.putText(result, all_classes[class_idx[i]], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        cv2.imwrite(f"CAM_{save_name}.jpg", result)

model._modules.get("layer4").register_forward_hook(hook_feature)

params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

img_original = cv2.imread('/home/david/Desktop/test/Echinaster sepositus_0958c.jpg')[:,:,::-1]
img = transformations(image=img_original)['image']
img = img.unsqueeze(0)
outputs = model(img)

probs = F.softmax(outputs, dim=1).data.squeeze()
class_idx = torch.argmax(probs).item()
print(probs)
CAMs = returnCAM(features_blobs[0], weight_softmax, [class_idx])

heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (img_original.shape[1], img_original.shape[0])), cv2.COLORMAP_JET)
result = heatmap * 0.1 + img_original * 0.9
cv2.putText(result, cfg.species[class_idx], (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

plt.imshow(result/255.)
plt.show()

