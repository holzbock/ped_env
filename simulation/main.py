#!/usr/bin/env python

# The base of this code is the 'start_recording.py' file of the CARLA examples.

# Copyright of the base:
#   Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
#   Barcelona (UAB).
#
#   This work is licensed under the terms of the MIT license.
#   For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import pdb

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import random
import time
import logging
import numpy as np
# import pygame

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sync_mode import CarlaSyncMode


# important_keypoints = [(1, 'crl_hips__C'), (2, 'crl_spine__C'), (3, 'crl_spine01__C'), (4, 'crl_shoulder__L'), (5, 'crl_arm__L'), (6, 'crl_foreArm__L'), 
# (7, 'crl_hand__L'), (28, 'crl_neck__C'), (29, 'crl_Head__C'), (30, 'crl_eye__L'), (31, 'crl_eye__R'), (32, 'crl_shoulder__R'), (33, 'crl_arm__R'), 
# (34, 'crl_foreArm__R'), (35, 'crl_hand__R'), (56, 'crl_thigh__R'), (57, 'crl_leg__R'), (58, 'crl_foot__R'), (61, 'crl_thigh__L'), (62, 'crl_leg__L'), 
# (63, 'crl_foot__L')]

important_keypoints = [1, 2, 3, 4, 5, 6, 7, 29, 30, 31, 32, 33, 34, 35, 56, 57, 58, 61, 62, 63]
# Cityscapes color: https://github.com/carla-simulator/carla/blob/f14acb257ebf44c302b225b02080ac5f0eedcf7f/LibCarla/source/carla/image/CityScapesPalette.h#L19
color_pedestrians = np.array([220,20,60]) # Pedestrian color in cityscapes 


# from: carla PythonAPI/examples/draw_skeleton.py
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


# based on: carla PythonAPI/examples/draw_skeleton.py
def get_screen_points(world2cam, K, points3d):
    
    # build the points array in numpy format as (x, y, z, 1) to be operable with a 4x4 matrix
    points = np.concatenate([np.stack(points3d), np.ones((len(points3d),1))], axis=1).T
    # convert world points to camera space
    points_camera = np.dot(world2cam, points)
    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    points_ = np.array([points_camera[1], points_camera[2] * -1, points_camera[0]])
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, points_)
    # normalize the values and transpose
    points_2d_ = np.array([points_2d[0, :] / points_2d[2, :], points_2d[1, :] / points_2d[2, :], points_2d[2, :]]).T
    # format x, y, z; remove z
    points_2d = points_2d_[:,:2]

    return points_2d


def get_log_dir(root):
    # get all existing dirs
    dirs = os.walk(root)
    dirs = [dir for dir in dirs]
    dirs = dirs[0][1]
    # get number for next log dir
    numbers = [-1,]
    for dir in dirs:
        if 'exp' in dir:
            num = int(dir.replace('exp', ''))
            numbers.append(num)
    last = max(numbers)
    # Make new log dir
    new = last + 1
    log_dir = os.path.join(root, 'exp' + str(new)) + '/'
    os.mkdir(log_dir)

    return log_dir


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '-f', '--recorder_filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-t', '--recorder_time',
        metavar='T',
        default=0,
        type=int,
        help='recorder duration (auto-stop)')
    args = argparser.parse_args()

    actor_list = []
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Get output dir
    log_dir = get_log_dir('./output')
    os.mkdir(os.path.join(log_dir,'rgb'))
    os.mkdir(os.path.join(log_dir,'sem_seg'))
    os.mkdir(os.path.join(log_dir,'skel'))
    os.mkdir(os.path.join(log_dir,'bbox'))
    os.mkdir(os.path.join(log_dir,'data'))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        world = client.get_world()
        bp_vehicle = [veh for veh in world.get_blueprint_library().filter('vehicle.*')]

        spawn_points = world.get_map().get_spawn_points()
        print('found %d spawn points.' % len(spawn_points))

        # Setup ego vehicle
        bp_ego_car = world.get_blueprint_library().find('vehicle.audi.etron')
        ego_car = world.spawn_actor(bp_ego_car, spawn_points[0])
        ego_car.set_autopilot(True)

        # setup cameras
        img_w = 1600
        img_h = 900
        fov = 64.5
        # place camera https://github.com/carla-simulator/carla/issues/1636#issuecomment-492188535
        rel_trans = carla.Transform(carla.Location(x=0.6, z=1.5))

        # Setup RGB camera
        bp_camera = world.get_blueprint_library().find('sensor.camera.rgb')
        # camera attributes https://carla.readthedocs.io/en/0.9.12/ref_sensors/#rgb-camera
        bp_camera.set_attribute('fov', str(fov))
        bp_camera.set_attribute('image_size_x', str(img_w))
        bp_camera.set_attribute('image_size_y', str(img_h))
        camera = world.spawn_actor(bp_camera, rel_trans, attach_to=ego_car)
        cam_intr_matr = build_projection_matrix(img_w, img_h, fov)

        # setup segmentation camera
        bp_seg_cam = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp_seg_cam.set_attribute('fov', str(fov))
        bp_seg_cam.set_attribute('image_size_x', str(img_w))
        bp_seg_cam.set_attribute('image_size_y', str(img_h))
        sem_camera = world.spawn_actor(bp_seg_cam, rel_trans, attach_to=ego_car)

        # setup depth camera
        bp_depth_cam = world.get_blueprint_library().find('sensor.camera.depth')
        bp_depth_cam.set_attribute('fov', str(fov))
        bp_depth_cam.set_attribute('image_size_x', str(img_w))
        bp_depth_cam.set_attribute('image_size_y', str(img_h))
        depth_camera = world.spawn_actor(bp_depth_cam, rel_trans, attach_to=ego_car)

        # setup gnss sensor
        bp_gnss = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0,0,0)
        gnss_rotation = carla.Rotation(0,0,0)
        gnss_transform = carla.Transform(gnss_location,gnss_rotation)
        ego_gnss = world.spawn_actor(bp_gnss, gnss_transform, attach_to=ego_car, attachment_type=carla.AttachmentType.Rigid)

        # Setup pedestrians
        walkers = list()
        walker_controllers = list()
        bp_walker = [w for w in world.get_blueprint_library().filter('walker.pedestrian.*')]
        walker_ai_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i, sp in enumerate(spawn_points):
            if i == 80:
                break
            walker = world.try_spawn_actor(random.choice(bp_walker), sp)
            if walker is not None:
                ai_controller = world.spawn_actor(walker_ai_controller_bp, carla.Transform(), walker)
                ai_controller.start()
                ai_controller.go_to_location(world.get_random_location_from_navigation())
                ai_controller.set_max_speed(1 + random.random())  # Between 1 and 2 m/s (default is 1.4 m/s).
                walkers.append(walker)
                walker_controllers.append(ai_controller)
            else:
                print('Pedestrian not spawned')      

        # Run simulation
        with CarlaSyncMode(world, camera, sem_camera, depth_camera, ego_gnss, fps=10) as sync_mode:
            while True:
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_sem, image_depth, gnss_data = sync_mode.tick(timeout=5.0)

                # get the pedestrian bones
                world2camera_matr = np.array(camera.get_transform().get_inverse_matrix())
                bones = list()
                points_all = list()
                for walker in walkers:
                    bones.append(walker.get_bones())
                    bone_index = list()
                    points = list()
                    for i, bone in enumerate(walker.get_bones().bone_transforms):
                        bone_index.append((i, bone.name))
                        points.append(np.array([bone.world.location.x, bone.world.location.y, bone.world.location.z]))
                    points_all.append(points)
                    
                # choose pedestrians which are in the image
                points_2d = list()
                points_3d = list()
                for points in points_all:
                    point2d = get_screen_points(world2camera_matr, cam_intr_matr, points)
                    if point2d[:,0].max() > 0 and point2d[:,0].min() < img_w:
                        points_2d.append(point2d)
                        points_3d.append(np.stack(points))

                # choose specific keypoints
                pos_3d = list()
                for i in range(len(points_2d)):
                    pos_3d.append(points_3d[i][0])
                    points_2d[i] = points_2d[i][important_keypoints]
                    points_3d[i] = points_3d[i][important_keypoints]

                # check if keypoint is in image
                for i in range(len(points_2d)):
                    bigger_x = np.round(points_2d[i][:,0]) < img_w
                    smaller_x = np.round(points_2d[i][:,0]) >= 0
                    bigger_y = np.round(points_2d[i][:,1]) < img_h
                    smaller_y = np.round(points_2d[i][:,1]) >= 0
                    # keep = bigger & smaller
                    points_2d[i][np.invert(smaller_x),0] = 0
                    points_2d[i][np.invert(bigger_x),0] = img_w - 1
                    points_2d[i][np.invert(smaller_y),1] = 0
                    points_2d[i][np.invert(bigger_y),1] = img_h - 1

                # choose pedestrians that are visible
                # convert carla image to numpy image
                image_sem.convert(carla.ColorConverter.CityScapesPalette)
                np_img_sem = np.frombuffer(image_sem.raw_data, dtype=np.uint8)
                np_img_sem = np.reshape(np_img_sem, (image_sem.height, image_sem.width, 4))
                np_img_sem = np_img_sem[:, :, :3] #  Take only RGB
                np_img_sem = np_img_sem[:, :, ::-1] # BGR

                visible_2d, visible_3d, pos_3d_visible = list(), list(), list()
                for i, pers2d in enumerate(points_2d):
                    it = 0
                    for p2d in pers2d:
                        color = np_img_sem[int(p2d.round()[1]), int(p2d.round()[0]),:]
                        if np.all(color == color_pedestrians):
                            it +=1
                    if it > 5:
                        visible_2d.append(points_2d[i])
                        visible_3d.append(points_3d[i])
                        pos_3d_visible.append(pos_3d[i])

                # sort out pedestrians that are behind a building 
                # but in the semantic of an other pedestrian
                image_depth.convert(carla.ColorConverter.Raw)
                np_img_depth = np.frombuffer(image_depth.raw_data, dtype=np.uint8)
                np_img_depth = np.reshape(np_img_depth, (image_depth.height, image_depth.width, 4))
                np_img_depth = np_img_depth[:, :, :3] #  Take only RGB
                np_img_depth = np_img_depth[:, :, ::-1] # BGR # needed else the distance is to far
                normalized = (np_img_depth[:,:,0].astype(np.int32) + np_img_depth[:,:,1].astype(np.int32) * 256 + \
                              np_img_depth[:,:,2].astype(np.int32) * 256 * 256) / (256 * 256 * 256 - 1)
                depth_meters = normalized * 1000
                inv_intr_matr = np.linalg.inv(cam_intr_matr)
                pix_img_y = np.tile(np.arange(depth_meters.shape[0]).reshape(-1,1), depth_meters.shape[1])
                pix_img_x = np.tile(np.arange(depth_meters.shape[1]).reshape(-1,1), depth_meters.shape[0]).T
                pix_img = np.stack([pix_img_x, pix_img_y], axis=2).reshape(-1,2)
                pix_img = np.concatenate([pix_img, np.ones((pix_img.shape[0],1))], axis=1)
                dm_cam_coord = (np.dot(inv_intr_matr, pix_img.T) * depth_meters.reshape(-1)).T
                dm_cam_coord = np.array([dm_cam_coord[:, 2], dm_cam_coord[:, 0], dm_cam_coord[:, 1]*-1]).T
                dm_cam_coord = np.concatenate([dm_cam_coord, np.ones((dm_cam_coord.shape[0],1))], axis=1)
                dm_world_coord = np.dot(np.array(camera.get_transform().get_matrix()), dm_cam_coord.T).T[:,:3]
                dm_world_coord = dm_world_coord.reshape(depth_meters.shape[0], depth_meters.shape[1], 3)

                visible_close_2d, visible_close_3d, pos_3d_visible_close = list(), list(), list()
                ego_pos = np.array([camera.get_transform().location.x, camera.get_transform().location.y])
                for i, (vis3d, vis2d) in enumerate(zip(visible_3d, visible_2d)):
                    dist = np.sqrt(np.sum((vis3d[0,:2]-ego_pos)**2))
                    coords = vis2d[0].astype(int)
                    dm_coord = dm_world_coord[coords[1], coords[0]]
                    dist_dm = np.sqrt(np.sum((dm_coord[:2]-ego_pos)**2))
                    # dist_dm should be always farer away then dist 
                    diff = np.abs(dist_dm - dist)
                    if diff < 1:
                        visible_close_2d.append(vis2d)
                        visible_close_3d.append(vis3d)
                        pos_3d_visible_close.append(pos_3d_visible[i])

                # get 2d bbox
                bbox_2d = list()
                for pos2d in visible_close_2d:
                    box = np.array([pos2d[:,0].min(), pos2d[:,1].min(), pos2d[:,0].max(), pos2d[:,1].max()])
                    bbox_2d.append(box)

                # Prepare gnss output data
                gnss_out_data = dict()
                gnss_out_data['xyz'] = np.array([gnss_data.transform.location.x, gnss_data.transform.location.y, gnss_data.transform.location.z])
                gnss_out_data['yaw_pitch_roll'] = np.array([gnss_data.transform.rotation.yaw, gnss_data.transform.rotation.pitch, gnss_data.transform.rotation.roll])

                # save data
                frame = image_rgb.frame
                image_rgb.save_to_disk(os.path.join(log_dir, 'rgb', 'img_%i.png'%frame))
                image_sem.save_to_disk(os.path.join(log_dir, 'sem_seg', 'img_%i.png'%frame))
                # plot skeletons
                img = Image.open(os.path.join(log_dir, 'rgb', 'img_%i.png'%frame))
                img = np.array(img)
                plt.close('all')
                fig, ax = plt.subplots()
                ax.grid(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.imshow(img)
                for points in visible_close_2d:
                    ax.plot(points[:,0], points[:,1], 'r.', ms=2)
                plt.savefig(os.path.join(log_dir, 'skel', 'img_%i.png'%frame), bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close('all')
                fig, ax = plt.subplots()
                ax.grid(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.imshow(img)
                for box in bbox_2d:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    x = box[0] + 0.5 * w
                    y = box[1] + 0.5 * h
                    patch = patches.Rectangle((box[0],box[1]), w, h, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(patch)
                plt.savefig(os.path.join(log_dir, 'bbox', 'img_%i.png'%frame), bbox_inches='tight', pad_inches=0, dpi=300)
                # save data (2d skeleton, 3d skeleton, ego position, intrinsic camera data, camera2world)
                out_data = dict()
                out_data['2d_skel'] = visible_close_2d
                out_data['3d_skel'] = visible_close_3d
                out_data['pos_3d'] = pos_3d_visible_close
                out_data['bbox_2d'] = bbox_2d
                out_data['ego_pos'] = gnss_out_data
                out_data['intr_cam'] = cam_intr_matr
                out_data['cam2world'] = np.array(camera.get_transform().get_matrix())
                out_data['world2cam'] = np.array(world2camera_matr)
                np.save(os.path.join(log_dir, 'data', 'data_%i'%frame), out_data)
                
    finally:
        # Destroy camera
        if camera.destroy():
            print('Camera destroyed.')
        else:
            print('Camera not destroyed.')
        
        # Destroy semantic camera
        if sem_camera.destroy():
            print('Semantic Camera destroyed.')
        else:
            print('Semanic Camera not destroyed.')

        # Destroy depth camera
        if depth_camera.destroy():
            print('Depth Camera destroyed.')
        else:
            print('Depth Camera not destroyed.')

        # Destroy ego car
        if ego_gnss.destroy():
            print('GNSS destroyed.')
        else:
            print('GNSS not destroyed.')
        
        # Destroy ego car
        if ego_car.destroy():
            print('Ego car destroyed.')
        else:
            print('Ego car not destroyed.')

        # Destroy controller for walkers
        controller_destroyed = list()
        for controller in walker_controllers:
            controller.stop()
            controller_destroyed.append(controller.destroy())
        if np.all(np.array(controller_destroyed)):
            print('All AI controller for the walkers destroyed.')
        else:
            print('Not all AI controller for the walkers destroyed.')

        # Destroy walkers
        walker_destroyed = list()
        for walker in walkers:
            walker_destroyed.append(walker.destroy())
        if np.all(np.array(controller_destroyed)):
            print('All walkers destroyed.')
        else:
            print('Not all walkers destroyed.')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
