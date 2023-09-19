# Code in this file is partially reused from mmpose https://github.com/open-mmlab/mmpose/tree/537bd8e543ab463fb55120d5caaa1ae22d6aaf06
import torch
import numpy as np
from copy import deepcopy
from mmpose.models.detectors import CID
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate
from mmpose.core.post_processing import oks_nms
import time

# basis: HRnet_W32
# from mmpose
cid_model_cfg = dict(
    type='CID',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
    ),
    keypoint_head=dict(
        type='CIDHead',
        in_channels=480,
        gfd_channels=32,
        num_joints=17,
        multi_hm_loss_factor=1.0,
        single_hm_loss_factor=4.0,
        contrastive_loss_factor=1.0,
        max_train_instances=200,
        prior_prob=0.01),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=17,
        flip_test=False,
        max_num_people=30,
        detection_threshold=0.01,
        center_pool_kernel=3))


class PersonDetection():
    def __init__(self, device='cuda:0', kp_thres=0.01):
        self.model_cfg = cid_model_cfg
        self.device = device
        self.kp_thres = kp_thres
        self.load_model()
        self.inference_time = 0
        self.data_loading_cfg = [{'type': 'BottomUpGetImgSize', 'test_scale_factor': [1]}, 
                            {'type': 'BottomUpResizeAlign', 'transforms': 
                            [{'type': 'ToTensor'}, {'type': 'NormalizeTensor', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}]}, 
                            {'type': 'Collect', 'keys': ['img'], 'meta_keys': ['image_file', 'aug_data', 'test_scale_factor', 'base_size', 'center', 'scale', 'flip_index']}]
        self.pipeline = Compose(self.data_loading_cfg)

        self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], 
                            [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], 
                            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.data = {
            'dataset': 'coco',
            'ann_info': {
                'image_size': np.array(512),
                'num_joints': 17,
                'image_file': 'tmp.jpg',
                'flip_index': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
                'skeleton': self.skeleton,
                }
            }
        
        self.sigmas = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])

        self.pose_kpt_color = np.array([[ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0]])
        self.pose_link_color = np.array([[  0, 255,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [255, 128,   0],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [  0, 255,   0],
                                        [255, 128,   0],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255],
                                        [ 51, 153, 255]])

    def load_model(self):
        self.model = CID(self.model_cfg['backbone'], self.model_cfg['keypoint_head'],
                   self.model_cfg['train_cfg'], self.model_cfg['test_cfg'])
        # state dict loaded from: https://mmpose.readthedocs.io/zh_CN/latest/papers/algorithms.html#cid-cvpr-2022
        state_dict = torch.load('CID_state_dict.pth')['state_dict']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()


    def inference(self, img, out_name=None):
        start = time.time()

        # prepare data
        data = deepcopy(self.data)
        data['img'] = img
        data = self.pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data['img_metas'] = data['img_metas'].data[0]
        data['img'] = data['img'].to(self.device)

        pose_results = list()
        with torch.no_grad():
            result = self.model(img=data['img'], img_metas=data['img_metas'], return_loss=False)

        # postprocess data
        # from: https://github.com/open-mmlab/mmpose/blob/21aeeb455fe198083cd3a16de9c9fc00bade8706/mmpose/apis/inference.py#L523
        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            if result['scores'][idx] > self.kp_thres:
                pose_results.append({
                    'keypoints': pred[:, :3],
                    'score': result['scores'][idx],
                    'area': area,
                })

        # pose nms
        # from: https://github.com/open-mmlab/mmpose/blob/21aeeb455fe198083cd3a16de9c9fc00bade8706/mmpose/apis/inference.py#L534
        keep = oks_nms(pose_results, 0.8, self.sigmas, score_per_joint=False)
        pose_results = [pose_results[_keep] for _keep in keep]
        self.inference_time = (time.time() - start) * 1000

        # plot results in image
        if out_name is not None:
            img_plot = self.plot(pose_results, img=data['img'].detach().cpu().numpy().squeeze(), out_name=out_name)
        else:
            img_plot = None

        return pose_results, img_plot
    

    def plot(self, pose_results, img, out_name):
        plot = list()
        for pose in pose_results:
            plot.append({'keypoints': pose['keypoints']})

        img_plot = self.model.show_result(img, 
                            plot, out_file=out_name, 
                            kpt_score_thr=self.kp_thres, 
                            pose_kpt_color=self.pose_kpt_color, 
                            pose_link_color=self.pose_link_color,
                            skeleton=self.skeleton)

        return img_plot