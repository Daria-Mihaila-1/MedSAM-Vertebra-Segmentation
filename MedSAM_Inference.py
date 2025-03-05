# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""
import tkinter as tk
# %% load environment
import cv2
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

from VertebraDataset import VertebraDatasetMedSAM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import matplotlib
from scipy.spatial.distance import directed_hausdorff
from torchmetrics.classification import BinaryJaccardIndex
matplotlib.use('TkAgg')
# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)

parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="assets/img_demo.png",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=str,
    default='[95, 255, 300, 350]',
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cpu", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_segmentation_results(boxes, masks, preds):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    for idx in range(len(boxes)):
        show_box(boxes[idx], ax[0])
        show_mask(masks[idx], ax[0])
    ax[0].set_title("Input Image and Bounding Box")

    # MedSAM Segmentation
    ax[1].imshow(image)
    for it, pred in enumerate(preds):
        color = 255 * show_mask(pred, ax[1], random_color=True)
    for box in boxes:
        show_box(box, ax[1])
    ax[1].set_title(f"MedSAM Segmentation with dice score")
    plt.show()


def select_file():
    # Open file dialog and store selected file path
    root = tk.Tk()
    root.mainloop()
    file_path = filedialog.askopenfilename()
    if file_path:
        return file_path
    root.destroy()
    # Update label with the selected file path


def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice


def separate_dice_coefficient(preds, targets):
    assert preds.shape == targets.shape
    smooth = 1.0
    dice = []
    for it, pred in enumerate(preds):
        intersection = (pred * targets[it]).sum()
        dice.append((2.0 * intersection + smooth) / (pred.sum() + targets[it].sum() + smooth))
    return dice


# TODO : make medsam work with batches --> compute dice loss + save images
#  compute other losses --> save them to excel
#  vezi si la IMedSAM cum se face --> do the same

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def custom_collate_fn(batch):
    # Separate images and targets
    images = torch.stack([item[0] for item in batch])
    masks = [item[1]['masks'] for item in batch]

    max_masks = max([mask.shape[0] for mask in masks])
    padded_masks = torch.stack([
        torch.cat(
            [mask, torch.zeros((max_masks - mask.shape[0], mask[0].shape[0], mask[0].shape[0]), device=mask.device,
                               dtype=mask.dtype)], dim=0)
        for mask in masks],
        dim=0)  # pad masks so that each image looks like it has the same number of masks add [0,0,0,0] masks

    # masks:
    # mask1 =-->torch.Tensor(16, 1024, 1024)
    # mask2 --> torch.Tensor(18, 1024, 1024)
    # tensors: in masks
    # mask1 --> (torch.Tensor(1024, 1024), torch.Tensor(1024, 1024) ... x 16

    # labels = [item[1]['labels'] for item in batch]
    # bboxes:
    # bbox1 --> list of 16
    #
    bboxes = [torch.tensor(item[1]['boxes']) for item in batch]
    max_bboxes = max(bbox.shape[0] for bbox in bboxes)
    padded_bboxes = torch.stack([
        torch.cat([bbox, torch.zeros((max_bboxes - bbox.shape[0], 4), device=bbox.device)], dim=0)
        for bbox in bboxes],
        dim=0)  # pad bboxes so that each image looks like it has the same number of bboxes add [0,0,0,0] bboxes

    img_file_names = [item[2] for item in batch]

    return images, padded_masks, padded_bboxes, img_file_names


def custom_collate_fn_inference(batch):
    # Separate images and targets
    images = torch.stack([item[0] for item in batch])
    masks = [item[1]['masks'] for item in batch]
    bboxes = [torch.tensor(item[1]['boxes']) for item in batch]

    img_file_names = [item[2] for item in batch]

    return images, masks, bboxes, img_file_names


device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# TODO incearca inferenta pe setul tau de date si calculeaza loss urile
img_np = io.imread(args.data_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024


def hausdorff_distance(pred, mask):
    # preds coords
    preds_coords = np.column_stack(np.where(pred == 1))
    targets_coords = np.column_stack(np.where(mask == 1))

    hausdf_pred_target = directed_hausdorff(preds_coords, targets_coords)[0]
    hausdf_target_pred = directed_hausdorff(targets_coords, preds_coords)[0]
    hausdf = max(hausdf_pred_target, hausdf_target_pred)
    return hausdf


def hausdorff_distance_img(preds, masks):
    hausdorff_distances = [hausdorff_distance(preds[idx], masks[idx]) for idx in range(len(preds))]
    return hausdorff_distances


with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    annotations_file_path = 'data/annotations_test.json'
    directory = 'scoliosis2-1/test'
    size = 1024
    masks_save_dir = "data/masks_test"
    transform = A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA),  # Resize the longer side to 1024
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, p=1.0),  # Pad to 1024x1024
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),  # Convert image and masks to PyTorch tensors
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))  # Bounding box params
    test_dataset = VertebraDatasetMedSAM(annotations_file_path,
                                         directory=directory,
                                         resize=size,
                                         transform=transform, masks_save_dir=masks_save_dir)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    sum_dice_score = 0
    metrics = {'dice score': [], 'mean_hausdorff_distance': [], 'img_file_name': []}

    for it, (images, masks, bboxes, file_names) in enumerate(test_dataloader):

        _, H, W = images[0].shape
        image_tensor = images  # (B, 3, 1024, 1024) the 0th dimension is added to resemble batch
        assert image_tensor.dtype == torch.float32
        assert torch.all(
            torch.logical_and(image_tensor >= 0, image_tensor <= 1)), "Image values must be in the range [0, 1]"
        bbox = bboxes[0]
        masks = masks[0]

        image_embedding = medsam_model.image_encoder(image_tensor)  # (8, 256, 64, 64)
        # TODO: write code for trying out the whole dataset here
        #  you have to:
        #  preprocess each image --> done by dataloader already
        #  put every image through the image embedder add the no_grad do this with the boxes
        #  (for each box)
        #  box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
        #  transfer box_np t0 1024x1024 scale
        #  box_1024 = box_np / np.array([W, H, W, H]) * 1024
        #  put each image with its boxes through the medsam_inference
        #  compute the dice loss for each resulted mask
        preds = []
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]  # x2 = x1 + width
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]  # y2 = y1 + height
        for box in bbox:
            box = torch.as_tensor(box[None, :], dtype=torch.float64)
            medsam_seg = medsam_inference(medsam_model, image_embedding, box, H, W)
            preds.append(medsam_seg)
        preds = np.array(preds)

        # dice score for each individual mask
        separate_dice_scores = separate_dice_coefficient(torch.tensor(np.array(preds)), masks)

        # Dice score for whole image --> gain perspective on whole image's prediction
        dice_score = dice_coefficient(torch.tensor(preds), masks)
        dice_score_value = dice_score.item()
        hausdf = np.mean(hausdorff_distance_img(preds, masks))

        iou = BinaryJaccardIndex()
        iou = iou(torch.as_tensor(preds), torch.tensor(np.array(masks)))


        # %% visualize results
        image = images[0].permute(1, 2, 0)

        # show_segmentation_results(boxes=bbox, masks=masks, preds=preds)
        # Save image with its dice coefficient at the end
        os.makedirs(r"C:\Users\Daria\Documents\UTCN\Disertatie\SAM\MEDSAM\assets\segmentations", exist_ok=True)
        # image_scaled = (medsam_seg * 255).astype(np.uint8)

        metrics['dice score'].append(dice_score_value)
        metrics['img_file_name'].append(file_names[0])
        metrics['mean_hausdorff_distance'].append(hausdf)
        print(f"Total Dice Score: {dice_score}")
        print(f"Mean Hausdorff: {hausdf}")
        
    df = pd.DataFrame(metrics, columns=metrics.keys())
    df.to_csv(os.path.join("assets/segmentations", "metrics_" + os.path.basename(directory) + ".csv"))

medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, 1024, 1024)
io.imsave(
    join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    medsam_seg,
    check_contrast=False,
)

# %% visualize results
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img_3c)
# show_box(box_np[0], ax[0])
# ax[0].set_title("Input Image and Bounding Box")
# ax[1].imshow(img_3c)
# show_mask(medsam_seg, ax[1])
# show_box(box_np[0], ax[1])
# ax[1].set_title("MedSAM Segmentation")
# plt.show()
