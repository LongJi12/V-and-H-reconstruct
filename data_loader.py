import math
import os,  json, copy
from PIL import Image 
opj = os.path.join

import numpy as np
from tqdm import tqdm
import torch
from pytorch3d.io import load_obj

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode, Instances, Boxes, Keypoints


def apollo_3dpose_loader(data_path, eval=False):

    if not eval:
        # Load gt verts
        deform_path = 'apollo_deform'
        verts = {}
        for i in range(79):
            vert, _,  _ = load_obj(opj(deform_path,f'{i}.obj'))
            verts[i] = vert.unsqueeze(0)

    # label and img path
    label_path = opj(data_path, 'apollo_annot')
    img_path = opj(data_path,'images')
    # files = [i.split('.')[0] for i in os.listdir(img_path)]
    files = [i.split('.')[0] for i in os.listdir(img_path) if i.startswith("190000")]

    print('Loading the dataset')
    data = []
    for id, file in tqdm(enumerate(files), total=len(files)):
        image_path = os.path.join(img_path, file + '.jpg')
        with Image.open(image_path) as img:
                width, height = img.size
        # image setting
        new_labels = {}
        new_labels['file_name'] = image_path
        new_labels['height'] = height
        new_labels['width'] = width
        new_labels['image_id'] = file
        # new_labels['file_name'] = opj(img_path,file+'.jpg')
        # new_labels['height'] = 2710
        # new_labels['width'] = 3384
        # new_labels['image_id'] = file

        if not eval:
            # load label
            labels = json.load(open(opj(label_path, file+'.json')))
            annotations = []
            for label in labels:
            
                if 'car_id' in label:
                    # exclude noise data
                    if label['pose'][-1] > 300:
                        continue
                    # Load labels
                    annotation = {}
                    # load 2d labels
                    annotation['bbox'] = label['bbox']
                    annotation['bbox_mode'] = BoxMode.XYXY_ABS
                    annotation['category_id'] = label["car_id"]
                    annotation['keypoints'] = label['keypoints']
                    # load 3d labels
                    annotation['trans'] = label['pose'][3:]
                    annotation['rotate'] = label['pose'][:3]
                    # load verts
                    annotation['verts'] = verts[label['car_id']]
                else:
                    # Load person labels
                    annotation = {}
                    # load 2d labels
                    annotation['bbox'] = label['bbox']
                    annotation['bbox_mode'] = BoxMode.XYXY_ABS
                    annotation['category_id'] = label["category_id"]
                    annotation['keypoints_2d'] = label['joints2D']
                    annotation['keypoints_3d'] = label['joints3D']
                    # Additional person-specific labels
                    annotation['pose'] = label['pose']
                    annotation['beta'] = label['shape']
                    annotation['valid'] = label['valid']

                annotations.append(annotation)

            new_labels['annotations'] = annotations
        else:
            new_labels['annotations'] = []
        data.append(new_labels)

    return data
    
class apollo_mapper:
    def __init__(self, resize):
        self.resize = resize
    def __call__(self,dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
    
        #Load img
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        #Resize 
        auglist = [T.Resize(self.resize),T.RandomBrightness(0.7, 1.3),T.RandomFlip(prob=0.5)] 
        augs = T.AugmentationList(auglist)
        auginput = T.AugInput(image)
        transform = augs(auginput)
        image = torch.from_numpy(auginput.image.astype("float32").transpose(2, 0, 1)) 

        # Resize labels: (bbox, keypoints)
        annos = [
            transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]

        return {
        # create the format that the model expects
        "image": image,
        'width': dataset_dict["width"],
        'height': dataset_dict["height"],
        'filename': dataset_dict["image_id"],
        "instances": annotations_to_instances(annos, image.shape[1:])
        }


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_keypoints", "gt_rotate", "gt_trans"
    """
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    # print('classes',classes)

    if (classes == 80).any():
        # 类别为80时的处理方式
        target.gt_classes = classes

         # 类别为80时的处理方式
        kpts2d = [obj.get("keypoints_2d", []) for obj in annos]
        kpts2d = [kp for kp in kpts2d if kp]  # 去除空的关键点
        target.gt_keypoints2d = Keypoints(kpts2d) if kpts2d else None

        kpts3d = [obj.get("keypoints_3d", []) for obj in annos]
        kpts3d = [kp for kp in kpts3d if kp]  # 去除空的关键点
        target.gt_keypoints3d = Keypoints(kpts3d) if kpts3d else None

        pose = [obj.get('pose', []) for obj in annos]
        target.gt_pose = torch.tensor(pose, dtype=torch.float32)

        beta = [obj.get('beta', []) for obj in annos]
        target.gt_beta = torch.tensor(beta, dtype=torch.float32)

        valid = [obj.get('valid', []) for obj in annos]
        target.gt_valid = torch.tensor(valid, dtype=torch.float32)
       
        # print('target',target)
    else:
        # 类别为0-79时的处理方式
        target.gt_classes = classes
        
        rotate = [obj.get('rotate', []) for obj in annos]
        target.gt_rotate = torch.tensor(rotate, dtype=torch.float32)

        trans = [obj.get('trans', []) for obj in annos]
        target.gt_trans = torch.tensor(trans, dtype=torch.float32)

        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
        
        verts = [obj.get('verts', []) for obj in annos]
        if len(verts) > 0 and verts[0].numel() > 0:  # 确保 verts 不是空的
            target.gt_verts = torch.cat(verts, dim=0)
        else:
            target.gt_verts = torch.zeros(0, dtype=torch.float32)
        # print('target1',target)

    return target 



def apollo_eval_loader(dataset, batch_size = 8, steps = None, resize = (1355,1692)):

    auglist = [T.Resize(resize)]
    augs = T.AugmentationList(auglist)

    if steps is None:
        steps = math.ceil(len(dataset)/batch_size)

    for step in range(steps):
        batch = copy.deepcopy(dataset[batch_size*step:batch_size*(step+1)])
        
        outputs = []
        for b in batch:            

            # LOAD IMAGE
            image = utils.read_image(b["file_name"], format="BGR")

            # RESIZE IMAGE
            auginput = T.AugInput(image)
            transform = augs(auginput)
            image = torch.from_numpy(auginput.image.astype("float32").transpose(2, 0, 1)) 

            # RESIZE LABELS (BBOX, KEYPOINTS)
            # annos = [
            #     utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            #     for annotation in b['annotations']
            # ]

            output = {
                # create the format that the model expects
                "image": image,
                'width': b["width"],
                'height': b["height"],
                'filename': b["image_id"],
                # "instances": annotations_to_instances(annos, image.shape[1:])
            }

            outputs.append(output)
        
        yield outputs

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    
    # print('ann',annotation)
    if "trans" in annotation:
        keypoint_hflip_indices = list(range(66))
    else:
       
        keypoint_hflip_indices = list(range(14))

    # print('key',keypoint_hflip_indices)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)

    # 3D aug
    for t in transforms:
        if isinstance(t, T.HFlipTransform):
            if "car_id" in annotation:
                if "pose" in annotation:
                    annotation['trans'][0] *= -1
                    annotation['rotate'][1] *= -1

    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    return annotation


if __name__ == '__main__':
    d = apollo_3dpose_loader('../apollo/val')
    for d_ in d:
        apollo_mapper(d_)
