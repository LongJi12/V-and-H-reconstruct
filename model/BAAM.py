from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances,Keypoints
from detectron2.utils.events import get_event_storage

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.registry import Registry
from .HMR import Regressor
from  .SMPL import SMPL
import numpy as np
import model.roi_head

@META_ARCH_REGISTRY.register()
class BAAM(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.batch_size=4
        self.hmr = Regressor()
        SMPL_MODEL_DIR = './data/base_data'
        self.smpl = SMPL(SMPL_MODEL_DIR,
                         batch_size=self.batch_size,
                         create_transl=False).to(device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(device)
        self.backbone_size_divisibility = self.backbone.size_divisibility
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def process_features(self, features, gt_instances):
        person_features = []
        car_features = {level_name: [] for level_name in features.keys()}
        person_instances = []
        car_instances = []

        for level_name in features.keys():
            feature = features[level_name]
            for instance, feat in zip(gt_instances, feature):
                if (instance.gt_classes == 80).any():
                    p7_transformed = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).view(-1)
                    person_features.append(p7_transformed)
                    person_instances.append(instance)
                else:
                    car_features[level_name].append(feat)

        if person_features:
            person_features = torch.cat(person_features, dim=0)
            if person_features.ndim == 1:
                person_features = person_features.unsqueeze(0)
            if person_features.shape[-1] != 2048:
                person_features = torch.nn.functional.interpolate(person_features.unsqueeze(0), size=(2048,), mode='linear', align_corners=False).squeeze(0)
        else:
            person_features = None

        for level_name in car_features.keys():
            if car_features[level_name]:
                # 使用第一个特征图的大小作为基准
                target_size = car_features[level_name][0].shape[-2:]
                # 对所有特征图进行插值变换，使其大小一致
                car_features[level_name] = [torch.nn.functional.interpolate(f.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0) for f in car_features[level_name]]
                car_features[level_name] = torch.stack(car_features[level_name])
            else:
                car_features[level_name] = None

        return person_features, car_features, person_instances, car_instances



    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], proposals = None, train_3d= True, train_key =True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            del proposals
            return self.inference(batched_inputs)

        # preprocessing image
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print('instances: %s', gt_instances)
        # feed forward
        features = self.backbone(images.tensor)
        # print('features',features)
        person_features, car_features, person_instances, car_instances = self.process_features(features, gt_instances)
        losses = {}
        # 判断类别为80的实例是否存在
        if person_features is not None:
            outputs = self.hmr(person_features)
            # print('outputs',outputs)

            # 确保 outputs 是一个包含字典的列表
            if isinstance(outputs, list) and all(isinstance(o, dict) for o in outputs):
                for output in outputs:
                    gt_betas = gt_instances[0].gt_beta
                    gt_pose = gt_instances[0].gt_pose

                    gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
                    gt_vertices = gt_out.vertices

                    keypoint_2d_loss = self.keypoint_loss(output['kp_2d'], gt_instances[0].gt_keypoints2d, 1.0, 1.0)
                    keypoint_3d_loss = self.keypoint_3d_loss(output['kp_3d'], gt_instances[0].gt_keypoints3d, 1.0)
                    shape_loss = self.shape_loss(output['verts'], gt_vertices, 1.0)
                    smpl_losses = self.smpl_losses(output['theta'][:, 3:], output['theta'][:, :3], gt_pose, gt_betas, 1.0)
                    print('keypoint_2dloss',keypoint_2d_loss)
                    print('keypoint_3dloss',keypoint_3d_loss)
                    print('shape_loss',shape_loss)
                    print('smpl_loss',smpl_losses)
                    losses.update({'keypoint_loss': keypoint_2d_loss})
                    losses.update({'keypoint_3d_loss': keypoint_3d_loss})
                    losses.update({'shape_loss': shape_loss})
                    losses.update({'smpl_losses': smpl_losses})
       
        print('car_features',car_features)
        
        if any(car_features.values()):
            proposals, proposal_losses = self.proposal_generator(images, car_features, gt_instances)
            detector_losses = self.roi_heads(images, features, proposals, gt_instances, train_3d=train_3d, train_key=train_key)
            losses.update(proposal_losses)
            losses.update(detector_losses)

        # # losses 
        # losses = {}
        # losses.update(proposal_losses)
        # losses.update(detector_losses)
        
        return losses
    

    def quat_to_rotmat(self,quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """ 
        norm_quat = quat
        norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                            2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                            2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
        return rotMat  
      
    def batch_rodrigues(self,theta):
        """Convert axis-angle representation to rotation matrix.
        Args:
            theta: size = [B, 3]
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

        return self.quat_to_rotmat(quat)
        
    
    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        # 确保 gt_keypoints_2d 是一个 tensor
        if isinstance(gt_keypoints_2d, Keypoints):
            gt_keypoints_2d = gt_keypoints_2d.tensor

        # 确保 pred_keypoints_2d 和 gt_keypoints_2d 的形状兼容
        if pred_keypoints_2d.shape[1] != gt_keypoints_2d.shape[1]:
            print('pred_keypoints_2d',pred_keypoints_2d.shape)          #[1,49,2]
            print('gt_keypoints_2d',gt_keypoints_2d.shape)              #[1,14,2]
            print(f"Warning: Mismatch in keypoint numbers. Truncating or padding keypoints to match.")
            if pred_keypoints_2d.shape[1] > gt_keypoints_2d.shape[1]:
                pred_keypoints_2d = pred_keypoints_2d[:, :gt_keypoints_2d.shape[1], :]
            else:
                padding = torch.zeros((pred_keypoints_2d.shape[0], gt_keypoints_2d.shape[1] - pred_keypoints_2d.shape[1], pred_keypoints_2d.shape[2])).to(pred_keypoints_2d.device)
                pred_keypoints_2d = torch.cat((pred_keypoints_2d, padding), dim=1)

        if gt_keypoints_2d.dim() == 2:
            gt_keypoints_2d = gt_keypoints_2d.unsqueeze(0)

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        # 确保 gt_keypoints_3d 是一个 tensor
        if isinstance(gt_keypoints_3d, Keypoints):
            gt_keypoints_3d = gt_keypoints_3d.tensor

        # 确保 pred_keypoints_3d 和 gt_keypoints_3d 的形状兼容
        if pred_keypoints_3d.shape[1] != gt_keypoints_3d.shape[1]:
            print('pred_keypoints_3d',pred_keypoints_3d.shape)    #[1,49,2]
            print('gt_keypoints_3d',gt_keypoints_3d.shape)        #[1,14,3]
            print(f"Warning: Mismatch in keypoint numbers. Truncating or padding keypoints to match.")
            if pred_keypoints_3d.shape[1] > gt_keypoints_3d.shape[1]:
                pred_keypoints_3d = pred_keypoints_3d[:, :gt_keypoints_3d.shape[1], :]
            else:
                padding = torch.zeros((pred_keypoints_3d.shape[0], gt_keypoints_3d.shape[1] - pred_keypoints_3d.shape[1], pred_keypoints_3d.shape[2])).to(pred_keypoints_3d.device)
                pred_keypoints_3d = torch.cat((pred_keypoints_3d, padding), dim=1)

        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0 and gt_keypoints_3d.size(1) > 3:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = self.batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        print('pred_rotmat_valid',pred_rotmat_valid.shape)        #[1,1,82]
        print('gt_rotmat_valid',gt_rotmat_valid.shape)         #[1,1,24,3,3]
        print('pred_betas_valid',pred_betas_valid.shape)         #[1,1,3]
        print('gt_rotmat_valid',gt_rotmat_valid.shape)           #[1,1,24,3,3]
        if len(pred_rotmat_valid) > 0 and pred_rotmat_valid.shape == gt_rotmat_valid.shape:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
        

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training


        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)

        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None) 

        # post process
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone_size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def set_multi_gpu(self, device_num):
        self.backbone = nn.DataParallel(self.backbone, device_ids= device_num)
        self.proposal_generator = nn.DataParallel(self.proposal_generator, device_ids= device_num)
        self.roi_heads = nn.DataParallel(self.roi_heads, device_ids= device_num)