from persons import Person, Skelett
import numpy as np
from scipy.optimize import linear_sum_assignment
import torchvision
import torch
import time


class Tracking():
    def __init__(self, img_w=1600, img_h=900, caah=64.5, delta_t=1/12, simulated=False, giou_thr=-0.5):
        self.persons = list()
        self.delete_after_timesteps = 3
        self.color = np.array([0, 0, 0], dtype=np.uint8)
        self.giou_thr = giou_thr
        self.person_counter = 0
        self.img_w = img_w
        self.img_h = img_h
        self.caah = caah
        self.delta_t = delta_t
        self.simulated = simulated
        self.mean_time_tracking, self.idx_tracking_time, self.time_tracking_init = 0, 0, False
        self.mean_time_dist, self.idx_dist_time, self.time_dist_init = 0, 0, False

    def get_kps_color(self):
        if self.color[0] <= 155:
            self.color[0] += 100
        elif self.color[1] <= 155:
            self.color[1] += 100
        elif self.color[2] <= 155:
            self.color[2] += 100
        else:
            self.color = np.array([0, 0, 0], dtype=np.uint8)
        return self.color

    def add_new_timestep(self, skeletons, delta_yaw, current_time, ego_pos):
        t_start = time.time()
        self.current_time = current_time
        self.person = [person.compensate_ego_rotation(delta_yaw) for person in self.persons]
        old_bboxes = np.array([person.get_last_element_bbox_2d() for person in self.persons])
        new_bboxes = np.array([skelett.get_bbox_2d() for skelett in skeletons])
        order, new, missing = self.associate_persons(old_bboxes, new_bboxes)
        if self.time_tracking_init and len(skeletons) > 0:
            self.mean_time_tracking = (self.mean_time_tracking * self.idx_tracking_time + (time.time() - t_start)) / (self.idx_tracking_time + 1)
            self.idx_tracking_time += 1
        else:
            self.time_tracking_init = True
        if new != []:
            for id in new:
                color = self.get_kps_color()
                pers = Person(color=color.copy(), person_id=self.person_counter, img_w=self.img_w, 
                              img_h=self.img_h, caah=self.caah, delta_t=self.delta_t, 
                              simulated=self.simulated)
                self.person_counter += 1
                pers.add_skelett(skeletons[id])
                self.persons.append(pers)
        if order != []:
            for id_new, id_old in zip(*order):
                t_start = time.time()
                self.persons[id_old].add_skelett(skeletons[id_new])
                if self.time_dist_init:
                    self.mean_time_dist = (self.mean_time_dist * self.idx_dist_time + (time.time() - t_start)) / (self.idx_dist_time + 1)
                    self.idx_dist_time += 1
                else:
                    self.time_dist_init = True
        if missing != []:
            for id in missing:
                self.persons[id].add_skelett(Skelett(time=current_time, ego_pos=ego_pos, 
                                                     simulated=self.simulated))
        self.delete_persons()


    def associate_persons(self, old_bboxes, new_bboxes):
        # return of the function: 
        # order of the new ones, new persons, missing persons
        num_old = old_bboxes.shape[0]
        num_new = new_bboxes.shape[0]
        # when no old data is available the tracking is not needed
        if num_old == 0:
            return [], list(range(num_new)), []
        elif num_new == 0:
            return [], [], list(range(num_old))
        else:
            giou = torchvision.ops.generalized_box_iou(torch.from_numpy(old_bboxes), torch.from_numpy(new_bboxes)).numpy()
            # remove new samples from the giou
            max_cols = np.max(giou, axis=0)
            old = max_cols > self.giou_thr
            giou_old = giou[:, old]
            new_indices = list(np.argwhere(max_cols <= self.giou_thr).squeeze(axis=1))
            old_indices = np.argwhere(max_cols > self.giou_thr).squeeze(axis=1)
            num_new = giou_old.shape[1]
            # remove missing samples from the giou (row)
            max_rows = np.max(giou, axis=1)
            old_row = max_rows > self.giou_thr
            giou_old = giou_old[old_row, :]
            missing_indices = list(np.argwhere(max_rows <= self.giou_thr).squeeze(axis=1))
            old_indices_row = np.argwhere(max_rows > self.giou_thr).squeeze(axis=1)
            num_old = giou_old.shape[0]
            # assign the remaining samples
            row_ind, col_ind = linear_sum_assignment(giou_old * -1)  # row indizes are sorted (old), col mixed (new)
            col_ind = list(old_indices[col_ind])
            row_ind = list(old_indices_row[row_ind])
            if num_new == num_old:
                return [list(col_ind), list(row_ind)], [] + new_indices, [] + missing_indices  # col_ind is the order of the new persons regarding to the old ones   
            elif num_new > num_old:
                # the index of the new person is missing in col_ind.
                ind_new = list(set(range(0, new_bboxes.shape[0])) - set(col_ind))
                return [list(col_ind), list(row_ind)], ind_new, [] + missing_indices
            elif num_new < num_old:
                # the index of the missing person is in row_ind
                ind_miss = list(set(range(0, old_bboxes.shape[0])) - set(row_ind))
                return [list(col_ind), list(row_ind)], [] + new_indices, ind_miss # col_ind is the order of the new persons regarding to the old ones; index of the missing person is skiped
            else:
                print("Error in associate person!")

    def delete_persons(self):
        delete = list()
        for id, person in enumerate(self.persons):
            valid_skeletons = person.valid_skeletons()
            if len(valid_skeletons) > 3:
                if not valid_skeletons[-1] and not valid_skeletons[-2] and not valid_skeletons[-3]:
                    delete.append(id)
        
        for i in np.arange(len(delete)-1, -1, -1):
            del self.persons[delete[i]]