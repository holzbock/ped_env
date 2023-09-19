from nuscenes.nuscenes import NuScenes
import numpy as np
import argparse
import time
import torch
import torchvision

from nuscenes_data import NuSceneData
from simulated_data import SimulatedData
from cid_pipeline import PersonDetection
from persons import Skelett
from tracking import Tracking


def get_scene(number, nusc):
    for sc in nusc.scene:
        name = 'scene-%0.4i'%number
        if sc['name'] == name:
            return sc
    print('Possible scenes are: ', nusc.list_scenes())
    raise ValueError('Could not find the scene with number %i'%number)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nuscenes', choices=['nuscenes', 'simulated'])
    opt = parser.parse_args()

    if opt.data == 'nuscenes':
        interesting_scenes = [61,103,553,916,655,757,796]
        nusc = NuScenes(dataroot='./data/v1.0-mini/', version='v1.0-mini', verbose=True)
        print(nusc.list_scenes())
        img_w = 1600
        img_h = 900
        caah = 64.5
        delta_t = 1/12
        simulated = False
        giou_thr = -0.3
    else:
        interesting_scenes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        img_w = 1600
        img_h = 900
        caah = 64.5
        delta_t = 0.1
        simulated = True
        giou_thr = -0.3

    overall_mean_time, overall_nn_time, idx_time = 0, 0, 0
    overall_error, overall_perc_error, idx_error = 0, 0, 0
    overall_scene_error, overall_scene_perc_error = list(), list()
    overall_error_bins, overall_perc_error_bins = list(), list()
    for _ in range(200): # each bin is 1m distance
        overall_error_bins.append(list())
        overall_perc_error_bins.append(list())

    for scene_id in interesting_scenes:
        scene_error, scene_perc_error, idx_scene_error = 0, 0, 0
        print('Evaluate scene: %i'%scene_id)
        if opt.data == 'nuscenes':
            scene = get_scene(scene_id, nusc)
            data_iter = NuSceneData(scene, nusc)
            person_detector = PersonDetection(device='cuda:0', kp_thres=0.015)
        else:
            data_iter = SimulatedData('./data/simulated_dataset/scene%i'%scene_id)
        
        old_yaw = None
        annotations3d = None
        persons = list()
        track = Tracking(img_w=img_w, img_h=img_h, caah=caah, delta_t=delta_t, simulated=simulated, giou_thr=giou_thr)
        annotations2d = None
        gt_ann = None

        for idx, (img, pose, annotations3d, annotations2d, ego_pos, yaw, cam2world, world2cam, intr_cam, timestep) in enumerate(data_iter):
            print(idx)
            t_start = time.time()
            # Human pose estimation for nuScenes with CID
            if opt.data == 'nuscenes':
                t_start_pose = time.time()
                pose, _ = person_detector.inference(img)
                if idx != 0: # exclude initialization stuff
                    overall_nn_time = (overall_nn_time * idx_time + (time.time() - t_start)) / (idx_time + 1)

            # create new skeletons
            skeletons = list()
            for p in pose:
                skel = Skelett(time=timestep, intr_cam_matr=intr_cam, 
                            extr_matr=cam2world, extr_matr_inv=world2cam, extr_ego=None, 
                            extr_camera=None, ego_pos=ego_pos, img_w=img_w, img_h=img_h,
                            simulated=simulated)
                skel.add_2d(p['keypoints'], p['score'])
                skeletons.append(skel)
            
            # Add the data of the timestep to the environment model
            delta = old_yaw - yaw if old_yaw is not None else 0
            old_yaw = yaw
            t_start_dist = time.time()
            track.add_new_timestep(skeletons, delta, timestep, ego_pos)
            if idx != 0 and idx != 1: # exclude initialization stuff
                overall_mean_time = (overall_mean_time * idx_time + (time.time() - t_start)) / (idx_time + 1)
                idx_time += 1

            # get data for evaluation and plotting
            if idx > 0:
                bboxes_pred = list()
                pred_3d = list()
                for person in track.persons:
                    if person(-1).has_2d():
                        # Get 2D bbox
                        bboxes_pred.append(person(-1).get_bbox_2d())
                        # Get 3D position
                        if person(-1).filtered_xyz is None:
                            pred_3d.append(np.array([np.nan, np.nan]))
                        else:
                            pred_3d.append(person(-1).filtered_xyz[:2])
                bboxes_pred = np.stack(bboxes_pred) if len(bboxes_pred) > 0 else None
                errors_img = list()

            if annotations3d is not None and idx > 0:
                if bboxes_pred is not None and annotations2d is not None:
                    # assign the ground truth bbox
                    iou = torchvision.ops.box_iou(torch.from_numpy(annotations2d), torch.from_numpy(bboxes_pred)).numpy()
                    anns3d = annotations3d[:,:2]
                    pred3d = np.stack(pred_3d)[:,:2]
                    reshaped_anns3d = np.tile(anns3d.reshape(-1, 2, 1), pred3d.shape[0])
                    reshaped_preds3d = np.tile(pred3d.reshape(-1, 2, 1), anns3d.shape[0]).transpose(2,1,0)
                    dist_all = np.sqrt(((reshaped_anns3d - reshaped_preds3d)**2).sum(axis=1))
                    dist_all[iou<=0] = 1e10
                    # use the closest gt bbox with an overlap
                    indexes = np.argmin(dist_all, axis=0)
                    exists_gt = np.max(iou, axis=0) > 0
                    indexes = indexes[exists_gt]
                    bboxes_pred = bboxes_pred[exists_gt]
                    cor_anns = annotations2d[indexes]
                    cor_3d_anns = [annotations3d[i] for i in indexes]
                    gt_ann = [None for _ in exists_gt]
                    i = 0
                    for j, ex in enumerate(exists_gt):
                        if ex:
                            gt_ann[j] = cor_3d_anns[i]
                            i += 1
                    for i, (pred, ann) in enumerate(zip(pred_3d, cor_3d_anns)):
                        if np.any(np.isnan(pred)):
                            continue
                        # calculate the absoulte error
                        dist = (pred - ann[:2])
                        dist = np.sqrt(np.sum(dist**2))
                        overall_error = (overall_error * idx_error + dist) / (idx_error + 1)
                        scene_error = (scene_error * idx_scene_error + dist) / (idx_scene_error + 1)
                        errors_img.append(dist)

                        # calculate the relative error
                        dist_ego_obj = np.sqrt(np.sum((ann[:2] - ego_pos)**2))
                        perc_error = dist / dist_ego_obj * 100
                        overall_perc_error = (overall_perc_error * idx_error + perc_error) / (idx_error + 1)
                        scene_perc_error = (scene_perc_error * idx_scene_error + perc_error) / (idx_scene_error + 1)
                        idx_error += 1
                        idx_scene_error += 1

                        # error in correlation to the distance between the vehicle and the pedestrian
                        bin = int(dist_ego_obj)
                        overall_error_bins[bin].append(dist)
                        overall_perc_error_bins[bin].append(perc_error)
                        
                        print('Error:', dist, '  Percentual Error: ', perc_error)
                
        overall_scene_error.append((scene_error, scene_id, idx_scene_error))
        overall_scene_perc_error.append((scene_perc_error, scene_id, idx_scene_error))

    print('Overall error for the whole evaluation: %fm; overall percentual error %f; number persons: %i'%(overall_error, overall_perc_error, idx_error))
    for error, perc_error in zip(overall_scene_error, overall_scene_perc_error):
        if error[1] != perc_error[1]:
            raise ValueError('No identical scene between the absolute error and the relative error.')
        print('Overall error for the scene %i: %fm; percentual error: %f; number persons: %i'%(error[1], error[0], perc_error[0], error[2]))

    print('Overall Mean Time: %fs'%(overall_mean_time))
    print('Overall NN Time: %fs'%(overall_nn_time))
    print('Overall Tracking Time: %fs'%(track.mean_time_tracking))
    print('Overall Dist Time: %fs'%(track.mean_time_dist))
