from nuscenes.utils.geometry_utils import BoxVisibility
from PIL import Image, ImageEnhance
import os
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt
from copy import deepcopy
from geometric_utils import extrinsic_calib, quaternion2yaw


class NuSceneData():
    def __init__(self, scene, nusc):
        self.scene = scene
        self.nusc = nusc
        self.next_token = self.scene['first_sample_token']
        self.next_iter = True
        self.cam = 'CAM_FRONT'
        self.lidar = 'LIDAR_TOP'
        self.next_token_camera = 0
        self.next_token_lidar = 0
        self.init = True
        self.plot = False


    def get_cam4annotation(self, sample, ann_token):
        boxes, cam = [], []
        cams = [key for key in sample['data'].keys() if 'CAM' in key]
        for cam in cams:
            _, boxes, _ = self.nusc.get_sample_data(sample['data'][cam], box_vis_level=BoxVisibility.ANY,
                                                    selected_anntokens=[ann_token])
            if len(boxes) > 0:
                break  # We found an image that matches. Let's abort.
        assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                            'Try using e.g. BoxVisibility.ANY.'
        assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'
        return cam


    def render(self, img, boxes, camera_intrinsic, name='tmp.png'):
        plt.close('all')
        _, ax = plt.subplots(1, 1, figsize=(9, 16))
        ax.imshow(img)
        for box in boxes:
            c = np.array(self.nusc.colormap[box.name]) / 255.0
            box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))
        ax.axis('off')
        ax.set_aspect('equal')
        plt.savefig(os.path.join('./3d_boxes', name), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
    

    def convert_global2camera_coordinate_system(self, box, data):
        # First step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', data['ego_pose_token'])
        box.translate(-np.array(poserecord['translation']))
        box.rotate(Quaternion(poserecord['rotation']).inverse)

        # Second step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        return box


    def get_data_sample(self, camera_data):
        sample = self.nusc.get('sample', camera_data['sample_token'])
        lidar_token = sample['data'][self.lidar]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        img_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        img = Image.open(img_path)
        cal_sensor = self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])

        sample_ann_tokens = sample['anns']
        annotations2d = list()
        annotations3d = list()
        bbox = list()
        for ann_token in sample_ann_tokens:
            sample_annotation = self.nusc.get('sample_annotation', ann_token)
            if 'pedestrian' in sample_annotation['category_name']:
                sample_class = sample_annotation['category_name']
                cam = self.get_cam4annotation(sample, ann_token)
                if cam != self.cam:
                    continue
            else:
                continue
            box_global = Box(center=sample_annotation['translation'], size=sample_annotation['size'], 
                             orientation=Quaternion(sample_annotation['rotation']), name=sample_class)
            box_camera = self.convert_global2camera_coordinate_system(deepcopy(box_global), camera_data)
            box_pixel = view_points(box_camera.corners(), np.array(cal_sensor['camera_intrinsic']), normalize=True)
            box_2d_pixel = box_pixel[:2]
            bbox_2d = np.array([np.min(box_2d_pixel[0]), np.min(box_2d_pixel[1]), np.max(box_2d_pixel[0]), np.max(box_2d_pixel[1])])
            annotations2d.append({'label': sample_class, 'xyxy': bbox_2d})
            annotations3d.append({'label': sample_class, 'xyz': box_global.center, 
                                  'wlh': box_global.wlh, 'orientation': box_global.orientation})
            if self.plot:
                bbox.append(box_camera)
                self.render(img, bbox, np.array(cal_sensor['camera_intrinsic']), name=str(camera_data['timestamp'])+'.png')
            
        return annotations3d, annotations2d
    

    def __iter__(self):
        self.next_iter = True
        return self
    

    def __next__(self):
        if self.next_iter:
            if self.init:
                sample = self.nusc.get('sample', self.next_token)
                camera_token = sample['data'][self.cam]
                self.next_token = self.nusc.get('sample_data', sample['data'][self.cam])['next']
                self.init = False
                img_data = self.nusc.get('sample_data', camera_token)
                img_path = os.path.join(self.nusc.dataroot, img_data['filename'])
                img = Image.open(img_path)
                ego_motion = self.nusc.get('ego_pose', img_data['ego_pose_token'])
                time = img_data['timestamp']
                camera_calib = self.nusc.get('calibrated_sensor', img_data['calibrated_sensor_token'])
                if img_data['is_key_frame']:
                    annotations3d, annotations2d = self.get_data_sample(img_data)
                else:
                    annotations2d, annotations3d = None, None
            else:
                img_data = self.nusc.get('sample_data', self.next_token)
                img_path = os.path.join(self.nusc.dataroot, img_data['filename'])
                img = Image.open(img_path)
                ego_motion = self.nusc.get('ego_pose', img_data['ego_pose_token'])
                self.next_token = self.nusc.get('sample_data', self.next_token)['next']
                time = img_data['timestamp']
                camera_calib = self.nusc.get('calibrated_sensor', img_data['calibrated_sensor_token'])
                if self.next_token == '':
                    self.next_iter = False
                if img_data['is_key_frame']:
                    annotations3d, annotations2d = self.get_data_sample(img_data)
                else:
                    annotations2d, annotations3d = None, None

            enhance_contrast = ImageEnhance.Contrast(img)
            img = enhance_contrast.enhance(1.3)
            enhance_brightness = ImageEnhance.Brightness(img)
            img = enhance_brightness.enhance(1.3)
            img = np.array(img)[:, :, ::-1].copy()

            if annotations3d == []:
                annotations3d = None
            if annotations2d == []:
                annotations2d = None
            
            if annotations3d is not None:
                if len(annotations3d) > 0:
                    annotations3d = np.stack([box['xyz'] for box in annotations3d])
                    annotations2d = [an['xyxy'] for an in annotations2d]
                    annotations2d = np.stack(annotations2d) if len(annotations2d) > 0 else None
            
            extr_camera = extrinsic_calib(camera_calib)
            extr_ego = extrinsic_calib(ego_motion)
            cam2world = np.dot(extr_ego, extr_camera) # from camera to world
            extr_camera_inv = np.linalg.inv(extr_camera)
            extr_ego_inv = np.linalg.inv(extr_ego)
            world2cam = np.dot(extr_camera_inv, extr_ego_inv) # from world to camera
            ego_pos = extr_ego[:2, -1]
            intr_cam = camera_calib['camera_intrinsic']
            yaw = quaternion2yaw(Quaternion(ego_motion['rotation']))

            return img, None, annotations3d, annotations2d, ego_pos, yaw, cam2world, world2cam, intr_cam, time
        else:
            raise StopIteration