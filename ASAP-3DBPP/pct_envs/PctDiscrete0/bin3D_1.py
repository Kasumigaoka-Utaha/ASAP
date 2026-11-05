from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random

class PackingDiscreteV1(gym.Env):
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, data_name=None, load_test_data=False,
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,keep_prev=False,
                 LNES = 'EMS',
                 output_mask = False,
                 **kwags):

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder
        # self.num_future_box = num_future_box

        self.shuffle = shuffle
        self.bin_size = container_size
        self.size_minimum = np.min(np.array(item_set))
        self.setting = setting
        self.item_set = item_set
        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2
        
        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)

        self.box_list = []
        self.place_idx = 0

        self.test = load_test_data
        # Actions sent by the neural policy are nine-value placement nodes.
        self.action_space = gym.spaces.Box(low=0.0, high=float(max(container_size)), shape=(9,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.keep_prev = keep_prev
        if self.keep_prev:
            self.prev_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = LNES  # Leaf Node Expansion Schemes: EMS (recommend), EV, EP, CP, FC


    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    # Calculate space utilization inside a bin.
    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])

    def reset(self,box_list=None):
        self.packed = []
        self.space.reset()
        if box_list is not None:
            self.box_list = box_list
        self.place_idx = 0
        cur_observation = self.cur_observation()
        return cur_observation

    # Count and return all PCT nodes.

    def cur_observation(self):
        boxes = []
        leaf_nodes = []
        self.next_box = self.gen_next_box()
        if self.next_holder == 1:
            next_n_box = self.next_box
        else:
            len_box = self.box_list.shape[0]
            if self.next_holder >= len_box-self.place_idx:
                next_n_box = self.box_list[self.place_idx:]
            else:
                next_n_box = self.box_list[self.place_idx:self.place_idx+self.next_holder]

        # if self.test:
        if self.setting == 3: 
            self.next_den = self.next_box[3]
            if self.next_holder == 1:
                next_n_den = self.next_den
            else:
                next_n_den = next_n_box[:,3]
        else: 
            self.next_den = 1
            if self.next_holder == 1:
                next_n_den = self.next_den
            else:
                n = next_n_box.shape[0]
                next_n_den = np.ones(n)

        boxes.append(self.space.box_vec)
        leaf_nodes.append(self.get_possible_position())

        # next_box = sorted(list(self.next_box))
        if self.next_holder > 1:
            if self.next_holder >= len_box-self.place_idx:
                if self.keep_prev:
                    self.next_box_vec[len_box-self.place_idx+1:, :] = self.prev_box_vec[:self.place_idx-1,:]
                else:
                    self.next_box_vec[len_box-self.place_idx+1:, :] = 0

                self.next_box_vec[0:len_box-self.place_idx, 3:6] = next_n_box
                self.next_box_vec[0:len_box-self.place_idx, 0] = next_n_den
                self.next_box_vec[0:len_box-self.place_idx, -1] = 1

                if self.keep_prev:
                    self.prev_box_vec[self.place_idx-1,:] = self.next_box_vec[0,:]
                    self.prev_box_vec[self.place_idx-1,-1] = 0
            else:
                self.next_box_vec[0:len_box-self.place_idx, 3:6] = next_n_box
                self.next_box_vec[0:len_box-self.place_idx, 0] = next_n_den
                self.next_box_vec[0:len_box-self.place_idx, -1] = 1
        else:
            self.next_box_vec[:, 3:6] = next_n_box
            self.next_box_vec[:, 0] = next_n_den
            self.next_box_vec[:, -1] = 1

        return np.reshape(np.concatenate((*boxes, *leaf_nodes, self.next_box_vec)), (-1))

    # Generate the next item to be placed.
    def gen_next_box(self):
        len_box_list = len(self.box_list)
        if self.place_idx >= len_box_list:
            # print("warning: out of instance range")
            self.place_idx = self.place_idx % len_box_list
        next_box = self.box_list[self.place_idx]
        self.place_idx += 1
        return next_box

    # Detect potential leaf nodes and check their feasibility.
    def get_possible_position(self):
        if   self.LNES == 'EMS':
            allPostion = self.space.EMSPoint(self.next_box,  self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box,  self.setting)
        elif self.LNES == 'EP':
            allPostion = self.space.ExtremePoint2D(self.next_box, self.setting)
        elif self.LNES == 'CP':
            allPostion = self.space.CornerPoint(self.next_box, self.setting)
        elif self.LNES == 'FC':
            allPostion = self.space.FullCoord(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        tmp_list = []

        for position in allPostion:
            xs, ys, zs, xe, ye, ze = position
            x = xe - xs
            y = ye - ys
            z = ze - zs

            if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                tmp_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])
                leaf_node_idx += 1

            if leaf_node_idx >= self.leaf_node_holder: break

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = int(leaf_node[3] - leaf_node[0])
        y = int(leaf_node[4] - leaf_node[1])
        z = list(self.next_box)
        z.remove(x)
        z.remove(y)
        z = z[0]
        action = (0, int(leaf_node[0]), int(leaf_node[1]))
        next_box = (x, y, int(z))
        return action, next_box

    def step(self, action):
        # print(action)
        if len(action) != 3: action, next_box = self.LeafNode2Action(action)
        else: next_box = self.next_box

        idx = [action[1], action[2]]
        bin_index = 0
        rotation_flag = action[0]
        succeeded = self.space.drop_box(next_box, idx, rotation_flag, self.next_den, self.setting)

        if not succeeded:
            reward = 0.0
            done = True
            info = {'counter': len(self.space.boxes), 'ratio': self.space.get_ratio(),
                    'reward': self.space.get_ratio() * 10}
            return self.cur_observation(), reward, done, info

        ################################################
        ############# cal leaf nodes here ##############
        ################################################
        packed_box = self.space.boxes[-1]

        if  self.LNES == 'EMS':
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                           packed_box.lx + packed_box.x, packed_box.ly + packed_box.y,
                                           packed_box.lz + packed_box.z])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        reward = box_ratio * 10

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['reward'] = self.space.get_ratio() * 10
        return self.cur_observation(), reward, done, info

