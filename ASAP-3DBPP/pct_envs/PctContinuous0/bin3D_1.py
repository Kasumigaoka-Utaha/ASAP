from .space import Space
import numpy as np
import gym
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
import torch
import random


class PackingContinuousV1(gym.Env):
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, data_name=None, load_test_data=False,
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,keep_prev=False,
                 sample_from_distribution = True,
                 sample_left_bound = 0.1,
                 sample_right_bound = 0.5,
                 **kwags):

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder

        self.shuffle = shuffle
        self.bin_size = container_size
        if sample_from_distribution:
            self.size_minimum = sample_left_bound
            self.sample_left_bound = sample_left_bound
            self.sample_right_bound = sample_right_bound
        else: self.size_minimum = np.min(np.array(item_set))
        self.setting = setting
        self.item_set = item_set
        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2

        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)
        self.box_list = []
        self.place_idx = 0

        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)
            assert isinstance(self.box_creator, BoxCreator)

        self.sample_from_distribution = sample_from_distribution
        if load_test_data:
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data
        # Actions sent by the neural policy are nine-value placement nodes.
        self.action_space = gym.spaces.Box(low=0.0, high=float(max(container_size)), shape=(9,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.keep_prev = keep_prev
        if self.keep_prev:
            self.prev_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = 'EMS'  # Leaf Node Expansion Schemes: EMS

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
            allPostion = self.space.EMSPoint(self.next_box, self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        # print(len(allPostion))

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

        out = min(self.leaf_node_holder-len(tmp_list),len(allPostion)-len(tmp_list))

        if out != 0:
            print(out)

        if len(tmp_list) != 0:
            leaf_node_vec[0:len(tmp_list)] = np.array(tmp_list)

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0,1,2]
        for r in record:
            if abs(x - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        for r in record:
            if abs(y - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        z = self.next_box[record[0]]
        action = (0, leaf_node[0], leaf_node[1])
        next_box = (x, y, z)
        return action, next_box

    def step(self, action):
        if len(action) != 3: action, next_box = self.LeafNode2Action(action)
        else: next_box = self.next_box

        idx = [round(action[1], 6), round(action[2], 6)]
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

        if self.LNES == 'EMS':
            # print(packed_box.lx)
            # print(packed_box.x)
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                       round(np.array(packed_box.lx + packed_box.x).item(), 6),
                                       round(np.array(packed_box.ly + packed_box.y).item(), 6),
                                       round(np.array(packed_box.lz + packed_box.z).item(), 6)])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        # self.box_creator.drop_box()  # remove current box from the list
        # self.box_creator.generate_box_size()  # add a new box to the list
        reward = box_ratio * 10

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['reward'] = self.space.get_ratio() * 10
        return self.cur_observation(), reward, done, info



class PackingContinuousV2(gym.Env):
    def __init__(self,
                 setting,
                 container_size=(10, 10, 10),
                 item_set=None, data_name=None, load_test_data=False,
                 internal_node_holder=80, leaf_node_holder=50, next_holder=1, shuffle=False,keep_prev=False,
                 sample_from_distribution = True,
                 sample_left_bound = 0.1,
                 sample_right_bound = 0.5,
                 output_mask = False,
                 **kwags):

        self.internal_node_holder = internal_node_holder
        self.leaf_node_holder = leaf_node_holder
        self.next_holder = next_holder
        self.output_mask = output_mask

        self.shuffle = shuffle
        self.bin_size = container_size
        if sample_from_distribution:
            self.size_minimum = sample_left_bound
            self.sample_left_bound = sample_left_bound
            self.sample_right_bound = sample_right_bound
        else: self.size_minimum = np.min(np.array(item_set))
        self.setting = setting
        self.item_set = item_set
        if self.setting == 2: self.orientation = 6
        else: self.orientation = 2

        # The class that maintains the contents of the bin.
        self.space = Space(*self.bin_size, self.size_minimum, self.internal_node_holder)
        self.box_list = []
        self.place_idx = 0

        # Generator for train/test data
        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomBoxCreator(item_set)
            assert isinstance(self.box_creator, BoxCreator)

        self.sample_from_distribution = sample_from_distribution
        if load_test_data:
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data
        # Actions sent by the neural policy are nine-value placement nodes.
        self.action_space = gym.spaces.Box(low=0.0, high=float(max(container_size)), shape=(9,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height,
                                                shape=((self.internal_node_holder + self.leaf_node_holder + self.next_holder) * 9,))
        self.next_box_vec = np.zeros((self.next_holder, 9))
        self.keep_prev = keep_prev
        if self.keep_prev:
            self.prev_box_vec = np.zeros((self.next_holder, 9))

        self.LNES = 'EMS'  # Leaf Node Expansion Schemes: EMS

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
            allPostion = self.space.EMSPoint(self.next_box, self.setting)
        elif self.LNES == 'EV':
            allPostion = self.space.EventPoint(self.next_box, self.setting)
        else:
            assert False, 'Wrong LNES'

        if self.shuffle:
            np.random.shuffle(allPostion)

        leaf_node_idx = 0
        leaf_node_vec = np.zeros((self.leaf_node_holder, 9))
        feasible_list = []
        infeasible_list = []

        if self.output_mask:
            for position in allPostion:
                xs, ys, zs, xe, ye, ze = position
                x = xe - xs
                y = ye - ys
                z = ze - zs

                if self.space.drop_box_virtual([x, y, z], (xs, ys), False, self.next_den, self.setting):
                    feasible_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 1, 1])
                    leaf_node_idx += 1
                else:
                    infeasible_list.append([xs, ys, zs, xe, ye, self.bin_size[2], 0, 0, 1])

                if leaf_node_idx >= self.leaf_node_holder: break

            final_len = min(len(feasible_list)+len(infeasible_list),self.leaf_node_holder)


            if final_len != 0:
                leaf_node_vec[0:len(feasible_list)] = np.array(feasible_list)
                leaf_node_vec[len(feasible_list):final_len] = np.array(infeasible_list[0:final_len-len(feasible_list)])

        else:
            final_len = min(len(allPostion),self.leaf_node_holder)
            leaf_node_vec[:,0:5] = np.array(allPostion)[0:final_len,0:5]
            leaf_node_vec[:,5] = self.bin_size[2]
            leaf_node_vec[:,7:9] = 1

        return leaf_node_vec

    # Convert the selected leaf node to the placement of the current item.
    def LeafNode2Action(self, leaf_node):
        if np.sum(leaf_node[0:6]) == 0: return (0, 0, 0), self.next_box
        x = round(leaf_node[3] - leaf_node[0], 6)
        y = round(leaf_node[4] - leaf_node[1], 6)
        record = [0,1,2]
        for r in record:
            if abs(x - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        for r in record:
            if abs(y - self.next_box[r]) < 1e-6:
                record.remove(r)
                break
        z = self.next_box[record[0]]
        action = (0, leaf_node[0], leaf_node[1])
        next_box = (x, y, z)
        return action, next_box

    def step(self, action):
        if len(action) != 3: action, next_box = self.LeafNode2Action(action)
        else: next_box = self.next_box

        idx = [round(action[1], 6), round(action[2], 6)]
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

        if self.LNES == 'EMS':
            # print(packed_box.lx)
            # print(packed_box.x)
            self.space.GENEMS([packed_box.lx, packed_box.ly, packed_box.lz,
                                       round(np.array(packed_box.lx + packed_box.x).item(), 6),
                                       round(np.array(packed_box.ly + packed_box.y).item(), 6),
                                       round(np.array(packed_box.lz + packed_box.z).item(), 6)])

        self.packed.append(
            [packed_box.x, packed_box.y, packed_box.z, packed_box.lx, packed_box.ly, packed_box.lz, bin_index])

        box_ratio = self.get_box_ratio()
        # self.box_creator.drop_box()  # remove current box from the list
        # self.box_creator.generate_box_size()  # add a new box to the list
        reward = box_ratio * 10

        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['reward'] = self.space.get_ratio() * 10
        return self.cur_observation(), reward, done, info
