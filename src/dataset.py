from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
from data_utils import load_depth
import data_utils


class SaveStage1OutputDataset(Dataset):
    """
    dataset for validation

    args:
      gt_paths (list(str)) = path to lidar ground truth
      radar_numpy_paths (list(str)) = list of paths to radar_point_clouds
      patch_size (list (y,x)) = size of patches you want to input into the model
      image_dir_path (str) = directory which stores nuscenes images
      dataset_type (train/val/test) = whcih dataset to load
    """

    def __init__(self, gt_paths, radar_numpy_paths, patch_size, image_dir_path):
        super(SaveStage1OutputDataset, self).__init__()
        self.radar_numpy_paths = radar_numpy_paths
        self.patch_size = np.asarray(patch_size) # (y,x)
        self.pad_size = self.patch_size[1] // 2
        self.image_dir_path = image_dir_path
        self.gt_paths = gt_paths
        self.num_samples = len(gt_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        radar_data_path = self.radar_numpy_paths[index]
        gt_path = self.gt_paths[index]

        camera_image_name = os.path.split(gt_path)[1].replace('.png','.jpg')

        img = Image.open(os.path.join(self.image_dir_path, camera_image_name)).convert('RGB')
        img = np.asarray(img, np.uint8)

        # flipping the image to C,H,W format
        img = np.transpose(img, (2,0,1))

        gt = load_depth(gt_path)

        radar_points_this_frame = np.load(radar_data_path)
        radar_points_this_frame = np.float32(radar_points_this_frame)

        return radar_points_this_frame, img, gt, self.patch_size, self.pad_size, camera_image_name


class BinaryClassificationDataset(Dataset):
    """
    dataset

    args:
      ground_truth_paths : np.ndarray  containing all ground truth paths
      radar_points_paths : np.ndarray containing all radar points numpy paths
      image_dirpath : (str) directory where nuscenes images are stored
      data_dirpath: (str) directory where we store the data
      patch_size: size of patch we want to feed into the model
      epsilon : depth within which we want to find corresponding lidar points
    """

    def __init__(self, ground_truth_paths, radar_points_paths, image_dirpath, data_dirpath, patch_size):
        super(BinaryClassificationDataset, self).__init__()

        self.num_samples = len(ground_truth_paths)

        assert len(ground_truth_paths) == len(radar_points_paths)

        #path to gt pngs
        self.ground_truth_paths = ground_truth_paths

        # path to radar numpys
        self.radar_points_paths = radar_points_paths

        # (y,x)
        self.patch_size = patch_size
        self.pad_size = self.patch_size[1] // 2

        self.image_dirpath = image_dirpath
        self.data_dirpath = data_dirpath


    def __len__(self):
        return self.num_samples

    def sample(self, radar_points):
        '''
        Arg(s)

        radar_points (np.ndarray): N x 3 (x,y,z) for all radar points in the frame

        Return(s)

        radar_point (np.ndarray) : 1 x 3 (x,y,z) for the radar point to feed into the model

        '''
        n_points = radar_points.shape[0]
        index = np.random.randint(0, n_points)

        return radar_points[index, :]

    def __getitem__(self, index):

        # Get path to radar numpy array
        radar_points_path = os.path.join(self.data_dirpath, self.radar_points_paths[index])

        # Get path to lidar map
        ground_truth_path = os.path.join(self.data_dirpath, self.ground_truth_paths[index])

        # Filename should be something like: n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488017862404
        filename = os.path.splitext(os.path.basename(ground_truth_path))[0]

        # Get name of image: n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488017862404.jpg
        image_filename = filename + '.jpg'

        # Get path to image
        image_path = os.path.join(self.image_dirpath, image_filename)

        # load radar points
        radar_points = np.load(radar_points_path)

        radar_points = radar_points.astype(np.float32)

        radar_point = self.sample(radar_points)

        # Convert to float and normalize between 0 and 1
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, np.uint8)

        # Flipping the image to C,H,W format
        image = np.transpose(image, (2, 0, 1))

        ground_truth = load_depth(ground_truth_path)

        ground_truth = np.expand_dims(ground_truth, 0)

        ground_truth = ground_truth.astype(np.float32)

        return image, radar_point, ground_truth

class BinaryClassificationDatasetVal(Dataset):
    """
    dataset

    args:
      ground_truth_paths : np.ndarray  containing all ground truth paths
      radar_points_paths : np.ndarray containing all radar points numpy paths
      image_dirpath : (str) directory where nuscenes images are stored
      data_dirpath: (str) directory where we store the data
      patch_size: size of patch we want to feed into the model
      epsilon : depth within which we want to find corresponding lidar points
    """

    def __init__(self, ground_truth_paths, radar_points_paths, image_dirpath, data_dirpath, patch_size):
        super(BinaryClassificationDatasetVal, self).__init__()

        self.num_samples = len(ground_truth_paths)

        assert len(ground_truth_paths) == len(radar_points_paths)

        #path to gt pngs
        self.ground_truth_paths = ground_truth_paths

        # path to radar numpys
        self.radar_points_paths = radar_points_paths

        # (y,x)
        self.patch_size = patch_size
        self.pad_size = self.patch_size[1] // 2

        self.image_dirpath = image_dirpath
        self.data_dirpath = data_dirpath


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # Get path to radar numpy array
        radar_points_path = os.path.join(self.data_dirpath, self.radar_points_paths[index])

        # Get path to lidar map
        ground_truth_path = os.path.join(self.data_dirpath, self.ground_truth_paths[index])

        # Filename should be something like: n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488017862404
        filename = os.path.splitext(os.path.basename(ground_truth_path))[0]

        # Get name of image: n008-2018-08-28-16-16-48-0400__CAM_FRONT__1535488017862404.jpg
        image_filename = filename + '.jpg'

        # Get path to image
        image_path = os.path.join(self.image_dirpath, image_filename)

        # load radar points
        radar_points = np.load(radar_points_path)

        radar_points = radar_points.astype(np.float32)

        # Convert to float and normalize between 0 and 1
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, np.uint8)

        # Flipping the image to C,H,W format
        image = np.transpose(image, (2, 0, 1))

        ground_truth = load_depth(ground_truth_path)

        ground_truth = ground_truth.astype(np.float32)

        return image, radar_points, ground_truth, np.asarray(self.patch_size), self.pad_size

# Second stage stuff starts here

class SecondStageDataset(Dataset):
    """
    dataset for second stage

    args:
      num_samples (int)  = total samples
      image_list (list of str) = list of image paths
      radar_points (array of points np_int) = radar points of shape x,y,z
      shift_points (array of points np_int) = shift from ground truth points of shape x,y
      padding_size (int) = padding size for the image along the width
      num_points (int) = number of points the network outputs
    """

    def __init__(self, image_list, dataset_type):
        super(SecondStageDataset, self).__init__()
        self.num_samples = len(image_list)
        self.img_list = image_list
        self.dataset_type = dataset_type

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = self.img_list[index]

        image = Image.open(os.path.join('..', self.img_list[index]))

        img = Image.open(os.path.join('..', image_path)).convert('RGB')
        img = np.asarray(image, np.uint8)

        # flipping the image to C,H,W format
        img = np.transpose(img, (2,0,1))

        dir_path, file_name = os.path.split(image_path)
        dir_path = dir_path.replace('samples','ground_truth_{}')
        dir_path = dir_path.format(self.dataset_type)
        file_name = os.path.splitext(file_name)[0] + '.png'
        gt_input_path = os.path.join(dir_path, file_name)

        gt = load_depth(gt_input_path)

        output_radar_image_dir_path, output_radar_image_file_name = os.path.split(image_path)
        output_radar_dir_path = output_radar_dir_path.replace('samples','model_output_radar_depths_{}')
        output_radar_dir_path = output_radar_dir_path.format(data_type)
        output_radar_image_file_name = os.path.splitext(output_radar_image_file_name)[0] + '.png'

        output_radar_depth_output_path = os.path.join(output_radar_dir_path, output_radar_image_file_name)

        output_radar_depth = load_depth(output_radar_depth_output_path)

        return output_radar_depth, img, gt


def random_crop(inputs, shape, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width

    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    return outputs

class FusionNetDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) depth
        (3) response
        (4) ground truth

    Arg(s):
        image_paths : list[str]
            paths to images
        depth_paths : list[str]
            paths to depth maps
        response_paths : list[str]
            paths to response maps
        ground_truth_paths : list[str]
            paths to ground truth depth maps
    '''

    def __init__(self,
                 image_paths,
                 depth_paths,
                 response_paths,
                 ground_truth_paths,
                 shape=None,
                 random_crop_type=['none']):

        self.n_sample = len(image_paths)

        for paths in [depth_paths, response_paths, ground_truth_paths]:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.response_paths = response_paths
        self.ground_truth_paths = ground_truth_paths

        self.shape = shape

        self.do_random_crop = \
            self.shape is not None and all([x > 0 for x in self.shape])

        # Augmentation
        self.random_crop_type = random_crop_type

    def __getitem__(self, index):

        # Load image
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = np.asarray(image, np.uint8)
        image = np.transpose(image, (2, 0, 1))

        # Load depth
        depth = load_depth(self.depth_paths[index])
        depth = np.expand_dims(depth, axis=0)

        # Load response
        response = load_depth(self.response_paths[index])
        response = np.expand_dims(response, axis=0)

        # Load response
        ground_truth = load_depth(self.ground_truth_paths[index])
        ground_truth = np.expand_dims(ground_truth, axis=0)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image, depth, response, ground_truth] = random_crop(
                inputs=[image, depth, response, ground_truth],
                shape=self.shape,
                crop_type=self.random_crop_type)

        # Convert to float32
        image, depth, response, ground_truth = [
            T.astype(np.float32)
            for T in [image, depth, response, ground_truth]
        ]
        return image, depth, response, ground_truth

    def __len__(self):
        return self.n_sample
