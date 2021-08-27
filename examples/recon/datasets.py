import os

import soft_renderer.functional as srf
import torch
import numpy as np
import tqdm


class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None, num_views = 24):
        self.class_ids = class_ids
        self.class_ids_labels = {class_id: i for i, class_id in enumerate(class_ids)}
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732
        self.num_views = num_views 

        self.class_ids_map = class_ids_map

        self.images = []
        self.voxels = []
        self.labels = []

        # storing number of samples per class
        self.num_data = {}

        # storing starting position of each class
        self.pos = {}

        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            self.images.append(list(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            self.voxels.append(list(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = self.images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
            self.labels += [class_id] * self.images[-1].shape[0]

        self.images = np.ascontiguousarray(np.concatenate(self.images, axis=0).reshape((-1, 4, 64, 64)))
        self.voxels = np.ascontiguousarray(np.concatenate(self.voxels, axis=0))
    
    def __len__(self):
        # each object has 24 images (perspectives)
        return len(self.labels)
    
    def __getitem__(self, idx):
        viewpoint_ids = np.arange(self.num_views)
        distances = torch.ones(self.num_views).float() * self.distance
        elevations = torch.ones(self.num_views).float() * self.elevation
        viewpoints = srf.get_points_from_angles(distances, elevations,
                                                    -torch.from_numpy(viewpoint_ids).float() * 15)
        return torch.from_numpy(self.images[idx*self.num_views:idx*self.num_views + self.num_views].astype('float32') / 255.), viewpoints, torch.tensor(self.class_ids_labels.get(self.labels[idx]), dtype=torch.long)

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations,
                                                    -torch.from_numpy(viewpoint_ids).float() * 15)

        for i in range(data_ids.size // (batch_size * 24)):
            images = torch.from_numpy(
                self.images[data_ids[i * batch_size * 24:(i + 1) * batch_size * 24]].astype('float32') / 255.)
            voxels = torch.from_numpy(
                self.voxels[data_ids[i * batch_size * 24:(i + 1) * batch_size * 24] // 24].astype('float32'))
            yield images, voxels, viewpoints_all[i * batch_size * 24:(i + 1) * batch_size * 24]
