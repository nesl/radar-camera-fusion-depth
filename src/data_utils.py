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


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=False, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def load_depth_with_validity_map(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map and validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z, v

def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def load_response(path, multiplier=2**14, data_format='HW'):
    '''
    Loads a response map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : response map
    '''

    # Loads response map from 16-bit PNG file
    response = np.array(Image.open(path), dtype=np.float32)

    # Convert using encodering multiplier
    response = response / multiplier

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        response = np.expand_dims(response, axis=0)
    elif data_format == 'HWC':
        response = np.expand_dims(response, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return response

def save_response(response, path, multiplier=2**14):
    '''
    Saves a response map to a 16-bit PNG file

    Arg(s):
        response : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
    '''

    response = np.uint32(response * multiplier)
    response = Image.fromarray(response, mode='I')
    response.save(path)

def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates

    Arg(s):
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

    query_row_idx, query_col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij')

    query_coord = np.stack(
        [query_row_idx.ravel(), query_col_idx.ravel()], axis=1)

    Z = interpolator(query_coord).reshape([rows, cols])

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z