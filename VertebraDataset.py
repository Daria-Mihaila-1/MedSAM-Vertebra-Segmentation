from torch.utils.data import Dataset, DataLoader
import torch
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from re import I
import random
import torchvision
from torch.nn.functional import normalize

def load_roboflow_dataset():
    from roboflow import Roboflow
    rf = Roboflow(api_key="XtK9gpYgUkPuQwD3c0s1")
    project = rf.workspace("myshroom-dataset-preprocessing").project("scoliosis2-dvnfp-exsn0")
    version = project.version(1)
    dataset = version.download("coco")
    return dataset


class VertebraDatasetMedSAM(Dataset):

    def __init__(self, annotations_file_path, masks_save_dir, directory="/content/full_dataset", resize=None,
                 transform=None):

        self.directory = directory
        self.annotations_file_path = annotations_file_path
        self.resize_size = resize
        self.transform = transform
        self.masks_save_dir = masks_save_dir
        with open(self.annotations_file_path, 'r') as f:
            if annotations_file_path.split('.')[-1] == 'xml':
                self.imgs_data = load_xml(self.annotations_file_path)
            else:
                print(f"loading json from {self.directory}")
                self.imgs_data = load_json(self.annotations_file_path)

    def __len__(self):
        return len(self.imgs_data)

    def __getitem__(self, idx):

        if not os.path.exists(self.directory):
            print("Directory does not exist")
            return None

        img_data = self.imgs_data[idx]
        img_file_name = img_data['file_name']
        width, height = img_data['width'], img_data['height']

        if self.directory.split('_')[-1] == 'npy':
            img = np.load(os.path.join(self.directory, img_file_name.strip('jpg') + 'npy'))
        else:
            img = cv2.imread(os.path.join(self.directory, img_file_name))
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print(f"image {img_file_name} is None")

        if img is not None:

            # torch.tensor(gt2D[None, :, :]).long(),
            gts = np.array(img_data['annotations'])
            masks_path = img_data['masks_path'].split('/')
            masks_path = os.path.join("data", masks_path[-2], masks_path[-1])

            instance_masks_file = np.load(masks_path, allow_pickle=True)
            instance_masks = instance_masks_file['masks']

            obj_ids = []
            boxes = []
            for it, gt in enumerate(gts):
                bbox = gt['bbox']
                mask_id = gt['id']
                obj_ids.append(mask_id)
                boxes.append(bbox)
            if img_file_name == "sunhl-1th-28-Feb-2017-239-D-AP_jpg.rf.42859cf3d8e90e08181fdf1c40202d0f.jpg":
                print("instance masks nefpre:", len(instance_masks))
                max_mask_size = np.max([np.sum(mask) for mask in instance_masks])
                instance_masks = np.array([mask for mask in instance_masks if np.sum(mask) != max_mask_size])
                print("instance masks after:", len(instance_masks))
                # modify the bounding boxes too
                max_box_size = np.max([bbox[2] * bbox[3] for bbox in boxes])
                max_box_index = [index for index, bbox in enumerate(boxes) if bbox[2] * bbox[3] == max_box_size]
                boxes = np.array([bbox for bbox in boxes if bbox[2] * bbox[3] < max_box_size])
                # modify labels too
                obj_ids.pop(max_box_index[0])

            if self.resize_size is not None and self.transform is not None:
                # print(type(img))
                transformed = self.transform(
                    image=img,  # Input image
                    masks=instance_masks,  # List of binary masks (one per instance)
                    bboxes=boxes,  # List of bounding boxes (optional)
                    labels=obj_ids,  # List of labels (optional)
                )
                # print(f"transformed boxes: {transformed['bboxes']}")

                # to have the tensor in shape (3, H, W)

                # returneaza o imagine de 1024x1024
                # un tensor de gt --> un singur obiect
                # tensor cu bboxurile de rigoare
                # numele imaginii
                target = {
                    'masks': transformed['masks'],
                    'labels': transformed['labels'],
                    'boxes': transformed['bboxes']
                }
                img_1024 = transformed['image'].float()

                assert (
                        torch.max(img_1024) <= 1.0 and torch.min(img_1024) >= 0.0
                ), "image should be normalized to [0, 1]"

            else:
                img_1024 = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
                instance_masks = [torch.tensor(mask, dtype=torch.float32) for mask in instance_masks]
                target = {
                    'masks': instance_masks,
                    'labels': obj_ids,
                    'boxes': boxes
                }

            return (
                img_1024,
                target,
                img_file_name,
            )
        else:
            print(f"Image file with name {img_file_name} does not exist in current directory")


def rle2mask_xml(rle, width, height, label=1):
    decoded = np.zeros(width * height)  # create bitmap container
    decoded_idx = 0
    value = 0
    if isinstance(rle, str):
        rle = list(map(int, rle.split(', ')))

    for v in rle:
        decoded[decoded_idx:decoded_idx + v] = [int(value)] * int(v)
        decoded_idx += v
        value = 1 - value

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((height, width))  # reshape to image size

    return decoded


def rle2mask(rle: str, width, height, label=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if isinstance(rle, str):
        rle = [int(x.strip()) for x in rle.split(",")]
    rle = np.asarray(rle, dtype=np.int32)  # Ensure NumPy array
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(width * height, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape((height, width))


def small_2_full_mask(mask: np.ndarray, img_width, img_height, top_left_x, top_left_y):
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    mask_height, mask_width = mask.shape

    full_mask[top_left_y: top_left_y + mask_height, top_left_x:top_left_x + mask_width] = mask
    return full_mask


import xml.etree.ElementTree as ET


def load_xml(xml_file: str, mask_save_dir="/content/preprocessed_masks"):
    tree = ET.parse(xml_file)  # Replace 'file.xml' with your actual XML file path
    root = tree.getroot()
    imgs_data = []
    images = root.findall('image')
    os.makedirs(mask_save_dir, exist_ok=True)

    for image in images:
        img_data = {
            'width': int(image.attrib['width']),
            'height': int(image.attrib['height']),
            'file_name': image.attrib['name'],
            'id': int(image.attrib['id'])
        }
        img_name = img_data['file_name'].strip('.jpg')
        masks_path = os.path.join(mask_save_dir, f"{img_name}_masks.npy")
        img_data['masks_path'] = masks_path

        annotations_data = []
        masks = []
        if image.findall('mask'):
            for it, mask in enumerate(image.findall("mask")):
                mask_data = {
                    'id': img_data['id'] * 30 + it,
                    'size': [int(img_data['width']), int(img_data['height'])],
                    'image_id': img_data['id'],
                    'bbox': np.array([
                        int(mask.attrib['left']),
                        int(mask.attrib['top']),
                        int(mask.attrib['width']),
                        int(mask.attrib['height'])
                    ])
                }
                if not f"{img_name}_masks.npy" in os.listdir(mask_save_dir):

                    np_mask = rle2mask_xml(mask.attrib['rle'], width=mask_data['size'][0], height=mask_data['size'][1])
                    np_mask = small_2_full_mask(np_mask,
                                                img_width=img_data['width'],
                                                img_height=img_data['height'],
                                                top_left_x=int(mask.attrib['left']),
                                                top_left_y=int(mask.attrib['top']))
                    masks.append(np_mask)
                annotations_data.append(mask_data)

            if f"{img_name}_masks.npy" not in os.listdir(mask_save_dir):
                masks_array = np.stack(masks)
                np.save(masks_path, masks_array)

            img_data['annotations'] = annotations_data

            imgs_data.append(img_data)

    return imgs_data


def load_json(json_file):
    with open(json_file, 'r') as f:

        annotations_data = json.load(f)

        images = annotations_data['images']
        annotations = annotations_data['annotations']
        # The only label in my dataset
        category = annotations_data['categories'][0]['id']

        imgs_data = []

        for img in images:
            image_id = img['id']
            # img_data = img

            # Delete unnecessary entries
            img_data = {k: v for k, v in img.items() if k not in {'coco_url', 'license', 'flickr_url', 'date_captured'}}

            # get the masks for the current image
            filtered_annotations = list(filter(lambda x: x['image_id'] == image_id, annotations))
            if filtered_annotations:
                # img_data['annotations'] = []
                imgs_data.append(img_data)

        return imgs_data




def visualize_item(img_tensor, masks, figsize=(10, 23), bboxes=None):
    """
    Visualizes an image with overlaid masks, where each mask is colored based on its label.
    """

    # Define a random colormap for unique labels
    # unique_labels = np.unique(labels).tolist()
    colors = [[random.randint(50, 255) for _ in range(3)] for mask in masks]
    # Convert image tensor to NumPy
    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) → (H, W, C)
    img_np = (img_tensor * 255).clip(0, 255).astype(np.uint8) if img_tensor.max() <= 1 else img_tensor.clip(0,
                                                                                                            255).astype(
        np.uint8)

    # Create an overlay for masks
    mask_overlay = np.zeros_like(img_np, dtype=np.uint8)

    for i in range(len(masks)):

        mask = masks[i].cpu().numpy()  # Ensure (H, W) shape

        color = colors[i]

        # Apply mask color
        colored_mask = np.zeros_like(img_np, dtype=np.uint8)
        for j in range(3):  # Apply color to all three channels
            colored_mask[:, :, j] = mask * color[j]

        # Accumulate masks using np.maximum to ensure visibility
        mask_overlay = np.maximum(mask_overlay, colored_mask)

        # Draw provided bounding boxes
        y_indices, x_indices = np.where(mask > 0)

        if len(x_indices) == 0 or len(y_indices) == 0:  # If no nonzero pixels
            return None  # No bounding box possible

        # Get bounding box coordinates
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Compute width and height
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        # bbox = bboxes[i]
        # if bbox is not None:
        #   x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), color, thickness=2)

    # Merge the accumulated mask with the original image
    img_with_mask = cv2.addWeighted(img_np, 1.0, mask_overlay, 0.5, 1)

    # # Show the result
    plt.figure(figsize=figsize)
    plt.imshow(img_with_mask)
    plt.axis("off")
    plt.show()


def custom_collate_fn(batch):
    # Separate images and targets
    images = [item[0] for item in batch]
    masks = [item[1]['masks'] for item in batch]
    # labels = [item[1]['labels'] for item in batch]
    bboxes = [item[1]['boxes'] for item in batch]
    img_file_names = [item[2] for item in batch]


if __name__ == '__main__':
    load_roboflow_dataset()
    size = 1024
    start = time.time()
    # /content/drive/MyDrive/UTCN/Disertatie/generated_annotations.json /content/annotations.xml /content/drive/MyDrive/UTCN/Disertatie/generated_annotations_modified_bboxes.json
    annotations_file_path = 'data/generated_annotations_modified_bboxes.json'
    directory = 'data/scoliosis2.v1i.voc/train'
    transform = A.Compose([
        A.LongestMaxSize(max_size=size),  # Resize the longer side to 1024
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, p=1.0),  # Pad to 1024x1024
        ToTensorV2(),  # Convert image and masks to PyTorch tensors
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))  # Bounding box params
    medsam_dataset = VertebraDatasetMedSAM(annotations_file_path,
                                           train=True,
                                           directory=directory,
                                           resize=size,
                                           #  h5f='/content/drive/MyDrive/UTCN/Disertatie/train_images.h5',
                                           transform=transform)
    import time
    from tqdm import tqdm


    def custom_collate_fn(batch):
        # Separate images and targets
        images = [item[0] for item in batch]
        masks = [item[1]['masks'] for item in batch]
        # labels = [item[1]['labels'] for item in batch]
        bboxes = [item[1]['boxes'] for item in batch]
        img_file_names = [item[2] for item in batch]

        return images, masks, bboxes, img_file_names


    # Example DataLoader setup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tr_dataloader = DataLoader(medsam_dataset,
                               batch_size=8,
                               shuffle=True,
                               collate_fn=custom_collate_fn,
                               num_workers=2,
                               pin_memory=True if torch.cuda.is_available() else False
                               )
    i = 0
    for step, (image, masks, bboxes, img_file_name) in enumerate(tr_dataloader):
        # suntem in batchuri de 8 imagini

        # show the example
        # visualize_item(img_tensor=image[0], masks=masks[0], bboxes=bboxes[0])

        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.close("all")

    stop = time.time()

    print(f"Time taken: {stop - start}")

    # stop = time.time()
    # print(f"Time passed {stop - start}")
    print(len(medsam_dataset))
