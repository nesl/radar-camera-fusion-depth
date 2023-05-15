from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import image as Image_Loader
import os
import pickle
import numpy as np
from typing import NamedTuple
from scipy.interpolate import LinearNDInterpolator
from PIL import Image

class Data_Struct(NamedTuple):
    scene_id: int
    sample_idx: int
    image_path: str
    ground_truth_points: np.ndarray
    input_points: np.ndarray
    ground_truth_label_path: str
    ground_truth_depth_path: str

def save_depth(z, path):
  '''
  Saves a depth map to a 16-bit PNG file
  Args:
    z : numpy
      depth map
    path : str
      path to store depth map
  '''
  z = np.uint32(z*256.0)
  z = Image.fromarray(z, mode='I')
  z.save(path)

def load_depth(path):
    '''
    Loads a depth map from a 16-bit PNG file
    Args:
    path : str
      path to 16-bit PNG file
    Returns:
    numpy : depth map
    '''
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z/256.0
    z[z <= 0] = 0.0
    return z


class Data_Utilities():
    """
    data utilities

    args:
        data_pickle_file_name: path to the pickle file that stores all the image paths and radar points and grouth truth

    """

    def __init__(self, data_pickle_file_name):
        super(Data_Utilities, self).__init__()
        self.file_name = data_pickle_file_name

    def load_data(self):
        """
        output:
          radar_input_samples (array of points np_int) = shape (x,y,z). Points from the radar with incorrect x and y
          image_path_final (list of str) = list of image paths
          shift_input_samples (array of points np_int) = shift from ground truth points of shape x,y
          scene_id_final (int) = ID of the scene. Not used in the code now but required to find which scene an input belongs to
          sample_idx_final (int) = ID of the sample from which these inputs are derived. Not used now
        """
        with open(self.file_name, 'rb') as handle:
            data_dict = pickle.load(handle)

        scene_id = []
        sample_idx = []
        image_path = []
        shift = []
        input_points = []
        lidar_path = []
        lidar_label_path = []

        # what is inside data_dict
        for i in range(0,len(data_dict)):
            scene_id.append(data_dict[i][0].scene_id)
            sample_idx.append(data_dict[i][0].sample_idx)
            image_path.append(data_dict[i][0].image_path)
            shift.append(data_dict[i][0].ground_truth_points)
            input_points.append(data_dict[i][0].input_points)
            lidar_label_path.append(data_dict[i][0].ground_truth_label_path)
            lidar_path.append(data_dict[i][0].ground_truth_depth_path)


        scene_id_final = []
        sample_idx_final = []
        image_path_final = []
        lidar_path_final = []
        input_points_x = []
        input_points_y = []
        input_points_z = []
        shift_points_x = []
        shift_points_y = []
        input_points = np.asarray(input_points)
        radar_input_samples = []
        shift_input_samples = []

        # scene_id_final and sample_idx_final are redundant
        for i in range(0,len(scene_id)):
            for j in range(0,len(shift[i])):
                scene_id_final.append(scene_id[i])
                sample_idx_final.append(sample_idx[i])
                image_path_final.append(image_path[i])
                radar_input_samples.append(input_points[i][j])
                shift_input_samples.append(shift[i][j])
                lidar_path_final.append(lidar_label_path[i].format(j))

        scene_id_final = np.asarray(scene_id_final)
        sample_idx_final = np.asarray(sample_idx_final)

        return radar_input_samples, shift_input_samples, image_path_final, lidar_path_final, scene_id_final, sample_idx_final


    def load_data_val(self):
        """
        output:
          radar_input_samples (array of points np_int) = shape (x,y,z). Points from the radar with incorrect x and y
          image_path_final (list of str) = list of image paths
          scene_id_final (int) = ID of the scene. Not used in the code now but required to find which scene an input belongs to
          sample_idx_final (int) = ID of the sample from which these inputs are derived. Not used now
        """
        with open(self.file_name, 'rb') as handle:
            data_dict = pickle.load(handle)

        scene_id = []
        sample_idx = []
        image_path = []
        shift = []
        input_points = []
        lidar_path = []
        lidar_label_path = []

        # what is inside data_dict
        for i in range(0,len(data_dict)):
            scene_id.append(data_dict[i][0].scene_id)
            sample_idx.append(data_dict[i][0].sample_idx)
            image_path.append(data_dict[i][0].image_path)
            shift.append(data_dict[i][0].ground_truth_points)
            input_points.append(data_dict[i][0].input_points)
            lidar_label_path.append(data_dict[i][0].ground_truth_label_path)
            lidar_path.append(data_dict[i][0].ground_truth_depth_path)

        scene_id_final = np.asarray(scene_id)
        sample_idx_final = np.asarray(sample_idx)
#         radar_input_samples = np.asarray(radar_input_samples)
#         shift_input_samples = np.asarray(shift_input_samples)
#         print(radar_input_samples[0][0])

        return input_points, image_path, scene_id_final, sample_idx_final, shift



def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates
    Args:
    depth_map : np.float32
      H x W depth map
    validity_map : np.float32
      H x W depth map
    log_space : bool
      if set then produce in log space
    Returns:
    np.float32 : H x W interpolated depth map
    '''
    assert depth_map.ndim == 2 and validity_map.ndim == 2
    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]
    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)
    interpolator = LinearNDInterpolator(
        # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    Z = interpolator(query_coord).reshape([rows, cols])
    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0
    return Z