import numpy as np
from geometric_utils import view_points
from filterpy.kalman import KalmanFilter
from numba import njit

class Skelett():
    def __init__(self, img_w=1600, img_h=900, time=None, intr_cam_matr=None, extr_matr=None, 
                 extr_matr_inv=None, extr_ego=None, extr_camera=None, ego_pos=None, simulated=False):
        self.skelett2d = None # first x, second y; pixel coordinates
        self.prob2d = None
        self.mid_2d = None # mid point of the keypoints for the tracking
        self.skelett3d = None # world coordinates
        self.prob3d = None
        self.mid_3d = None
        self._3d = None
        self.origin = None
        self.norm_skel_2d = None # normalized pixel coordinates
        self.norm_skel_3d = None # normalized world coordinates to the mid of the hip
        self.score = 1
        self.img_w = img_w
        self.img_h = img_h
        self.time = time
        self.intr_cam_matr = np.array(intr_cam_matr)
        self.extr_matr = extr_matr
        self.extr_matr_inv = extr_matr_inv
        self.extr_ego = extr_ego
        self.extr_camera = extr_camera
        self.ego_pos = ego_pos
        self.xyz = None
        self.filtered_xyz = None
        self.wlh = None
        self.xy = None
        self.wh = None
        self.keep = None
        self.pred = None
        self.simulated = simulated

    def has_2d(self):
        return True if self.skelett2d is not None else False
        
    def has_3d(self):
        return True if self.skelett3d is not None else False
    
    def has_dist(self):
        return self._3d is not None and self.origin is not None
    
    def add_2d(self, skelett, score=None):
        assert skelett.shape[1] == 3, '2D skeletons must have 3 values. 2 for the dimension and one for the prob.'
        self.skelett2d = skelett[:, :2]
        self.prob2d = skelett[:, -1]
        self.mid_2d = self.skelett2d.mean(axis=0, keepdims=True)
        self.normalize2d()
        if self.intr_cam_matr is not None and self.extr_matr is not None:
            self.calculate_3d_all()

    def calculate_3d_all(self):
        self._3d_camera = (self.skelett2d - np.array([self.intr_cam_matr[0,2], self.intr_cam_matr[1,2]])) / np.array([self.intr_cam_matr[0,0], self.intr_cam_matr[1,1]])
        _3d = np.concatenate([self._3d_camera, np.ones((self._3d_camera.shape[0],1))], axis=1)
        _3d = normalize(_3d)
        # Undo the conversion from the simulation
        if self.simulated: # from cam2world
            _3d = np.array([_3d[:,2], _3d[:,0], _3d[:,1]*-1]).T
        self._3d = np.dot(self.extr_matr[:3,:3], _3d.T).T[:, :3] #.squeeze() # transforms to the world coordinates
        self.origin = np.dot(self.extr_matr, np.array([0,0,1,1]).reshape(1,4).T).T[:, :3].squeeze() # gives the origin point of the camera

    def add_3d(self, skelett):
        assert skelett.shape[1] == 4, '3D skeletons must have 4 values. 3 for the dimension and one for the prob.'
        self.skelett2d = skelett[:, :3]
        self.prob2d = skelett[:, -1]
        self.mid_point_3d = self.skelett3d.mean(axis=0)
        self.normalize3d()
    
    def get_2d(self):
        assert self.has_2d(), 'No 2d skelett available'
        return self.skelett2d

    def get_3d(self):
        assert self.has_3d(), 'No 3d skelett available'
        return self.skelett3 
    
    def get_bbox_2d(self):
        box = np.array([np.min(self.skelett2d[:,0]), np.min(self.skelett2d[:,1]), np.max(self.skelett2d[:,0]), np.max(self.skelett2d[:,1])])
        return box

    def normalize2d(self):
        # normalize 2d skelekton
        assert self.skelett2d.shape[-1] == 2
        self.norm_skel_2d = self.skelett2d/self.img_w*2 - [1, self.img_h/self.img_w]

    def normalize3d(self):
        # normalize 3d skeleton to the hip
        hip = self.skelett3d[2, :].view(self.skelett3d.shape[0], -1)
        self.norm_skel_3d = self.skelett3d - hip

    def get_norm_2d(self):
        assert self.has_2d(), 'No 2d skelett available'
        return self.norm_skel_2d 
        
    def get_norm_3d(self):
        assert self.has_3d(), 'No 3d skelett available'
        return self.norm_skel_3d
    
    def compensate_ego_rot(self, delta):
        if self.skelett2d is not None:
            self.skelett2d[:, 0] += delta
            self.mid_2d = self.skelett2d.mean(axis=0, keepdims=True)
            self.normalize2d()


# https://math.stackexchange.com/questions/13734/how-to-find-shortest-distance-between-two-skew-lines-in-3d
# from: https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
@njit
def line_intersection3d(o1, d1, o2, d2):
    # compute unit vectors of directions of lines 1 and 2
    d1 = (d1) / np.linalg.norm(d1)
    d2 = (d2) / np.linalg.norm(d2)
    # find unit direction vector for line 3, which is perpendicular to lines 1 and 2
    d3 = np.cross(d2, d1)
    d3 /= np.linalg.norm(d3)
    # solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
    rhs = o2 - o1
    lhs = np.zeros((3,3))
    lhs[:,0] = d1
    lhs[:,1] = -d2
    lhs[:,2] = d3
    out = np.linalg.solve(lhs, rhs)
    p1 = o1 + out[0] * d1
    p2 = o2 + out[1] * d2
    return (p1 + p2) / 2

@njit
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


@njit
def ref_ang(_3d, ego_pos, xyz):
    direction = np.zeros(2)
    direction[0] = _3d[:,0].mean()
    direction[1] = _3d[:,1].mean()
    direction = normalize(direction.reshape(1,-1))
    diff = xyz[:2] - ego_pos
    length = np.sqrt(np.sum(diff**2))
    new = ego_pos + direction * length
    return new


@njit
def refine_dist(xyz, pred, extr_matr_inv, intr_cam_matr, _2d, ego_pos, scale_rate, simulated=False):
    up_down = np.ones((2,4))
    up_down[0,0] = xyz[0]
    up_down[0,1] = xyz[1]
    up_down[0,2] = 0
    up_down[1,0] = xyz[0]
    up_down[1,1] = xyz[1]
    up_down[1,2] = 1.7 # estimate that the person is 1.7m
    pred_camera = np.dot(extr_matr_inv, up_down.T).T
    # Undo the conversion from the simulation
    if simulated: #world2cam
        tmp = np.zeros_like(pred_camera)
        tmp[:,0] = pred_camera[:, 1]
        tmp[:,1] = pred_camera[:, 2] * -1
        tmp[:,2] = pred_camera[:, 0]
        pred_camera = tmp
    img_coord = view_points(pred_camera[:, :3].T, intr_cam_matr, normalize=True)[:2].T
    original_bb_high = _2d[:,1].max() - _2d[:,1].min()
    estimated_bb_high = img_coord[0,1] - img_coord[1,1]
    scale = 1 + (estimated_bb_high / original_bb_high - 1) / scale_rate
    if scale > 20:
        scale = 20
    if scale < -20:
        scale = -20
    diff = xyz[:2] - ego_pos
    length = np.sqrt(np.sum(diff**2)) * scale
    norm = normalize(diff.reshape(1,-1))
    new = (ego_pos + norm * length)
    return img_coord, scale, new


@njit
def calculate_3d_all(_3d_1, prob2d_1, origin_1, _3d_2, prob2d_2, origin_2):
    pred_all = list()
    prob_all = list()
    for d1, d2, p1, p2 in zip(_3d_2, _3d_1, prob2d_2, prob2d_1):
        pred = line_intersection3d(origin_2, d1, origin_1, d2)
        pred_all.append(pred)
        prob_all.append(p1 * p2)
    return pred_all, prob_all


@njit
def get_prediction(pred, prob, extr_matr_inv):
    tmp = np.ones((pred.shape[0],4))
    tmp[:, :3] = pred
    pred_camera = np.dot(extr_matr_inv, tmp.T).T
    keep1 = pred_camera[:,2] > 0
    prob_thres = np.percentile(prob, 30)
    keep2 = prob >= prob_thres
    keep = keep1 & keep2
    if np.any(keep):
        pred = pred[keep]
        prob = prob[keep]
    pred_all = pred
    pred_out = np.zeros(3)
    pred_out[0] = pred[:,0].mean()
    pred_out[1] = pred[:,1].mean()
    pred_out[2] = pred[:,2].mean()
    prob = prob.mean()
    return pred_all, pred_out, prob, keep


class Person():
    def __init__(self, buffer_len=15, caah=64.5, img_w=1600, img_h=900, delta_t=1/12, color=None, person_id=0, simulated=False):
        self.buffer_len = buffer_len
        self.buffer = list()
        self.camera_aperature_angle_horizontal = caah
        self.img_w = img_w
        self.img_h = img_h
        self.delta_t = delta_t
        self.color = color
        self.filter = KalmanFilter(dim_x=4, dim_z=2)
        self.filter_initialized = False
        self.person_id = person_id
        self.simulated = simulated
        self.preinit_filter()

    def preinit_filter(self):
        # initialization example: https://www.cbcity.de/das-kalman-filter-einfach-erklaert-teil-2
        if not self.simulated:
            self.filter.F = np.array([[1, 0, self.delta_t, 0], [0, 1, 0, self.delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
            # Messmatrix
            self.filter.H = np.array([[1,0,0,0], [0,1,0,0]])
            # Kovarianzmatrix
            self.filter.P *= 5
            # Messrauschkovarianz
            sigma_x_m = 16
            sigma_y_m = 16
            self.filter.R = np.array([[sigma_x_m, 0], 
                                      [0, sigma_y_m]])
            # Prozessrauschkovarianz
            sigma_x = 2
            sigma_y = 2
            self.filter.Q = np.array([[0.25*self.delta_t**4*sigma_x, 0.5*self.delta_t**3*sigma_x, 0, 0],
                                      [0, 0, 0.25*self.delta_t**4*sigma_y, 0.5*self.delta_t**3*sigma_y],
                                      [0.5*self.delta_t**3*sigma_x, self.delta_t**2*sigma_x, 0, 0],
                                      [0, 0, 0.5*self.delta_t**3*sigma_y, self.delta_t**2*sigma_y]])
        else:
            self.filter.F = np.array([[1, 0, self.delta_t, 0], [0, 1, 0, self.delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
            # Messmatrix
            self.filter.H = np.array([[1,0,0,0], [0,1,0,0]])
            # Kovarianzmatrix
            self.filter.P *= 5
            # Messrauschkovarianz
            sigma_x_m = 3
            sigma_y_m = 3
            self.filter.R = np.array([[sigma_x_m, 0], 
                                      [0, sigma_y_m]])
            # Prozessrauschkovarianz
            sigma_x = 0.075
            sigma_y = 0.075
            self.filter.Q = np.array([[0.25*self.delta_t**4*sigma_x, 0.5*self.delta_t**3*sigma_x, 0, 0],
                                      [0, 0, 0.25*self.delta_t**4*sigma_y, 0.5*self.delta_t**3*sigma_y],
                                      [0.5*self.delta_t**3*sigma_x, self.delta_t**2*sigma_x, 0, 0],
                                      [0, 0, 0.5*self.delta_t**3*sigma_y, self.delta_t**2*sigma_y]])
            

    def __call__(self, idx):
        return self.buffer[idx]
    
    def get_last_filled_skelett(self,):
        if self.buffer[-1].has_2d():
            return self.buffer[-1]
        elif self.buffer[-2].has_2d():
            return self.buffer[-2]
        elif self.buffer[-3].has_2d():
            return self.buffer[-3]
        elif self.buffer[-4].has_2d():
            return self.buffer[-4]

    def add_skelett(self, skelett):
        assert isinstance(skelett, Skelett), 'New element is not of class Skelett.'
        if len(self.buffer) > 0:
            last_skel = self.get_last_filled_skelett()
            if skelett.has_dist():
                self.calculate_dist(skelett, last_skel)
                for i in range(15):
                    self.refine_dist(skelett, scale_rate=i+5)
                    self.refine_angle(skelett)
                if self.filter_initialized:
                    self.filter.predict()
                    self.filter.update(np.array([[skelett.xyz[0]], [skelett.xyz[1]]]))
                    skelett.filtered_xyz = self.filter.x.squeeze()[:3]
                else:
                    self.filter.x = np.array([[skelett.xyz[0]], [skelett.xyz[1]], [2], [2]])
                    skelett.filtered_xyz = self.filter.x.squeeze()[:3]
                    self.filter_initialized = True
        self.buffer.append(skelett)
        if len(self.buffer) > self.buffer_len:
            del self.buffer[0]        

    def refine_dist(self, skelett, scale_rate=2):
        skelett.img_coord, skelett.scale, skelett.xyz[:2] = refine_dist(skelett.xyz, skelett.pred, skelett.extr_matr_inv, 
                                            skelett.intr_cam_matr, skelett.get_2d(), skelett.ego_pos, scale_rate=scale_rate, 
                                            simulated=self.simulated)

    def refine_angle(self, skelett):
        skelett.xyz = ref_ang(skelett._3d, skelett.ego_pos, skelett.xyz).squeeze()

    def calculate_dist(self, skelett, last_skel):
        pred, prob = calculate_3d_all(skelett._3d, skelett.prob2d, skelett.origin, last_skel._3d, last_skel.prob2d, last_skel.origin)
        skelett.pred, skelett.xyz, skelett.prob, skelett.keep = get_prediction(np.stack(pred), np.stack(prob), skelett.extr_matr_inv)
    
    def calculate_3d_all(self, skelett1, skelett2):
        pred_all = list()
        prob_all = list()
        for d1, d2, p1, p2 in zip(skelett2._3d, skelett1._3d, skelett2.prob2d, skelett1.prob2d):
            pred = line_intersection3d(skelett2.origin, d1, skelett1.origin, d2)
            pred_all.append(pred)
            prob_all.append(p1 * p2)
        return pred_all, prob_all

    # from: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    def line_intersection2d(self, o1, d1, o2, d2):
        def cross_prod_2d(x, y):
            return x[0] * y[1] - x[1] * y[0]
        if np.abs(cross_prod_2d(d1, d2)) < 1e-6:
            return o1 * np.nan
        t = cross_prod_2d(o2 - o1, d2) / cross_prod_2d(d1, d2)
        return o1 + t * d1

    def get_element(self, index):
        return self.buffer[index]
    
    def get_last_element(self):
        return self.get_element(-1)
    
    def get_last_element_bbox_2d(self):
        if self.get_element(-1).has_2d():
            return self.get_element(-1).get_bbox_2d()
        elif self.get_element(-2).has_2d():
            return self.get_element(-2).get_bbox_2d()
        elif self.get_element(-3).has_2d():
            return self.get_element(-3).get_bbox_2d()
        else:
            raise ValueError('Calculating the mid for the last valid 2d skeleton is not possible.')
    
    def __len__(self):
        return len(self.buffer)
    
    def compensate_ego_rotation(self, delta_rotation):
        # delta- --> left; delta+ --> right
        delta_x = delta_rotation/self.camera_aperature_angle_horizontal * self.img_w
        delta_x *= -1 # *-1 because the car turns e.g. left but the object in the image moves to the right boarder
        for pose in self.buffer:
            pose.compensate_ego_rot(delta_x)

    def valid_skeletons(self):
        return [skel.has_2d() for skel in self.buffer]