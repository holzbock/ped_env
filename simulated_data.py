import numpy as np
import os
from PIL import Image

class SimulatedData():
    def __init__(self, path):
        self.path = path
        self.next_iter = True
        self.next = 0
        self.init = True
        self.plot = False
        files = [x for x in os.walk(os.path.join(self.path, 'data'))][-1][-1]
        names = [file.split('.')[0] for file in files]
        numbers = [int(name.split('_')[-1]) for name in names]
        self.start = min(numbers)
        self.stop = max(numbers)
        self.next = self.start

    
    def __iter__(self):
        self.next_iter = True
        return self
    

    def __next__(self):
        if self.next_iter:
            img = Image.open(os.path.join(self.path, 'rgb', 'img_%i.png'%self.next))
            img = np.array(img)
            data = np.load(os.path.join(self.path, 'data', 'data_%i.npy'%self.next), allow_pickle=True).item()
            pose = [{'keypoints': np.concatenate([p, np.ones((p.shape[0],1))], axis=1), 'score': 1.} for p in data['2d_skel']]
            annotations2d = np.stack(data['bbox_2d']) if data['bbox_2d'] != [] else None
            annotations3d = np.stack(data['pos_3d']) if data['pos_3d'] != [] else None
            ego_pos = data['ego_pos']['xyz']
            yaw = data['ego_pos']['yaw_pitch_roll'][0]
            cam2world = data['cam2world']
            world2cam = data['world2cam']
            intr_cam = data['intr_cam']
            time = self.next
            
            self.next += 1
            if self.next > self.stop:
                self.next_iter = False
            return img, pose, annotations3d, annotations2d, ego_pos[:2], yaw, cam2world, world2cam, intr_cam, time
        else:
            raise StopIteration