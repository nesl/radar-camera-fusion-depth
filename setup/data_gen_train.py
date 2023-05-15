from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, copy
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from matplotlib import image
from sklearn.neighbors import KDTree
from typing import NamedTuple
import pickle
from PIL import Image
import os.path as osp
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import multiprocessing as mp
from data_utils import save_depth


# Create the nuScene object
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/akash/Documents/nuScenes', verbose=True)
nusc_explorer = NuScenesExplorer(nusc)

max_scenes = 850

# Max number of threads to use for thread pool
n_thread = 28

# number of point closest to the radar input that you wnt in your ground truth. If the number of points is less than the number specified, we just copy the points again.
num_points = 100
n_forward = 9
n_backward = 9

# Euclidean distance that you want to search in. Higher value = more error in depth
KDTREE_QUERY_RADIUS = 0.4

# More weight to depth = more movement along x axis and less along z
KDTREE_DEPTH_WEIGHT = 1.0

pickle_file_name_template = "train_nuScenes_dataset_lidar_maps_interpolated_merged_{}_{}_{}_kdtree_radius_{}_depth_{}_with_filter.pkl"
pickle_file_name = pickle_file_name_template.format(n_forward, n_backward, num_points, KDTREE_QUERY_RADIUS, KDTREE_DEPTH_WEIGHT)

# https://stackoverflow.com/questions/35988/c-like-structures-in-python

class Data_Struct(NamedTuple):
    scene_id: int
    sample_idx: int
    image_path: str
    input_points: np.ndarray
    ground_truth_label_path: str
    ground_truth_depth_path: str


def get_train_val_split_ids():
    """
    given the nuscenes object, find out which scene ids correspond to which set. The split is taken from the official nuScene split available here: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py
    : return train_split_ids, test_split_ids : lists containing ids of the scenes that are in each split
    """

    train_file_name = "train_ids.pkl"
    val_file_name = "val_ids.pkl"

    open_file = open(train_file_name, "rb")
    train_ids = pickle.load(open_file)
    open_file.close()

    open_file = open(val_file_name, "rb")
    val_ids = pickle.load(open_file)
    open_file.close()

    if DEBUG:
        train_ids_final = [1]
        return train_ids_final, val_ids

    return train_ids, val_ids

def point_cloud_to_image(point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
        minimum_distance_from_camera : float32
            threshold for removing points that exceeds minimum distance from camera
    Returns:
        numpy[float32] : 2 x N array of x, y
        numpy[float32] : N array of z
        numpy[float32] : camera image
    """

    # Get dictionary of containing path to image, pose, etc.
    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image = Image.open(osp.join(nusc.dataroot, camera['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_lidar_to_body['translation']))

    # Second step: transform from ego to the global frame.
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pose_body_to_camera = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_body_to_camera['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_camera['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    coloring = point_cloud.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Points will be 3 x N
    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(coloring.shape[0], dtype=bool)
    mask = np.logical_and(mask, coloring > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)

    # Select points that are more than min distance from camera and not on edge of image
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, image

def camera_to_lidar_frame(point_cloud,
                          lidar_sensor_token,
                          camera_token):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
    Returns:
        PointCloud : nuScenes point cloud object
    """

    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    # First step: transform from camera into ego.
    pose_camera_to_body = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_camera_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_camera_to_body['translation']))

    # Second step: transform from ego vehicle frame to global frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global frame to ego frame
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform point cloud from body to lidar
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_lidar_to_body['translation']))
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix.T)

    return point_cloud

def merge_lidar_point_clouds2(nusc,
                              current_sample_token,
                              n_forward,
                              n_backward):
    """
    Merges Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
    Returns:
        numpy[float32] : 2 x N of x, y for lidar points projected into the image
        numpy[float32] : N depths of lidar points

    """

    nusc_explorer = NuScenesExplorer(nusc)

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_lidar_token = current_sample['data']['LIDAR_TOP']

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    # Project the lidar frame into the camera frame
    main_points_lidar, main_coloring_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=main_lidar_token,
        camera_token=main_camera_token)

    # Convert nuScenes format to numpy for image
    main_image = np.asarray(main_image)

    # Create an empty lidar image
    main_lidar_image = np.zeros((main_image.shape[0], main_image.shape[1]))

    # Get all bounding boxes for the lidar data in the current sample
    _, main_boxes, main_camera_intrinsic = nusc.get_sample_data(
        main_camera_token,
        box_vis_level=BoxVisibility.ANY,
        use_flat_vehicle_coordinates=False)

    main_points_lidar_quantized = np.round(main_points_lidar).astype(int)

    # Iterating through each lidar point and plotting them onto the lidar image
    for point_idx in range(0, main_points_lidar_quantized.shape[1]):
        # Get x and y index in image frame
        x = main_points_lidar_quantized[0, point_idx]
        y = main_points_lidar_quantized[1, point_idx]

        # Value of y, x is the depth
        main_lidar_image[y, x] = main_coloring_lidar[point_idx]

    # Create a validity map to check which elements of the lidar image are valid
    main_validity_map = np.where(main_lidar_image > 0, 1, 0)

    # Count forward and backward frames
    n_forward_processed = 0
    n_backward_processed = 0

    # Initialize next sample as current sample
    next_sample = copy.deepcopy(current_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud in `next' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the next sample amd move forward
        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample', next_sample_token)

        # Get lidar and camera token in the current sample
        next_lidar_token = next_sample['data']['LIDAR_TOP']
        next_camera_token = next_sample['data']['CAM_FRONT']

        # Get bounding box in image frame to remove vehicles from point cloud
        _, next_boxes, next_camera_intrinsics = nusc.get_sample_data(
            next_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map next frame point cloud to image so we can remove vehicle based on bounding boxes
        next_points_lidar, next_coloring_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=next_lidar_token,
            camera_token=next_camera_token)

        next_lidar_image = np.zeros_like(main_lidar_image)

        next_points_lidar_quantized = np.round(next_points_lidar).astype(int)
        # Plots depth values onto the image
        for idx in range(0, next_points_lidar_quantized.shape[-1]):
            x, y = next_points_lidar_quantized[0:2, idx]
            next_lidar_image[y, x] = next_coloring_lidar[idx]

        # Remove points in vehicle bounding boxes
        for box in next_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=next_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                next_lidar_image[min_y:max_y, min_x:max_x] = 0

        # Now we need to convert image format to point cloud array format (y, x, z)
        next_lidar_points_y, next_lidar_points_x  = np.nonzero(next_lidar_image)
        next_lidar_points_z = next_lidar_image[next_lidar_points_y, next_lidar_points_x]

        # Backproject to camera frame as 3 x N
        x_y_homogeneous = np.stack([
            next_lidar_points_x,
            next_lidar_points_y,
            np.ones_like(next_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(next_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(next_lidar_points_z, axis=0)

        # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
        # So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)

        # Convert lidar point cloud into a nuScene LidarPointCloud object
        next_point_cloud = LidarPointCloud(x_y_z)

        # Now we can transform the points back to the lidar frame of reference
        next_point_cloud = camera_to_lidar_frame(
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=next_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_lidar_main, next_coloring_lidar_main, _ = point_cloud_to_image(
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        next_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for idx in range(0, next_points_lidar_main_quantized.shape[-1]):
            x, y = next_points_lidar_main_quantized[0:2, idx]
            next_lidar_image_main[y, x] = next_coloring_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                next_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0

        # Convert image format to point cloud format
        next_lidar_points_main_y, next_lidar_points_main_x  = np.nonzero(next_lidar_image_main)
        next_lidar_points_main_z = next_lidar_image_main[next_lidar_points_main_y, next_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        next_points_lidar_main = np.stack([
            next_lidar_points_main_x,
            next_lidar_points_main_y],
            axis=0)
        next_coloring_lidar_main = next_lidar_points_main_z

        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for point_idx in range(0, next_points_lidar_main_quantized.shape[1]):
            x = next_points_lidar_main_quantized[0, point_idx]
            y = next_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_coloring_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = next_coloring_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = next_coloring_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1

    # Initialize previous sample as current sample
    prev_sample = copy.deepcopy(current_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud in `prev' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the previous sample and move sample backward
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample', prev_sample_token)

        # Get lidar and camera token in the previous sample
        prev_lidar_token = prev_sample['data']['LIDAR_TOP']
        prev_camera_token = prev_sample['data']['CAM_FRONT']

        # Get bounding box in image frame to remove vehicles from point cloud
        _, prev_boxes, prev_camera_intrinsics = nusc.get_sample_data(
            prev_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map next frame point cloud to image so we can remove vehicle based on bounding boxes
        prev_points_lidar, prev_coloring_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=prev_lidar_token,
            camera_token=prev_camera_token)

        prev_lidar_image = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_quantized = np.round(prev_points_lidar).astype(int)

        for idx in range(0, prev_points_lidar_quantized.shape[-1]):
            x, y = prev_points_lidar_quantized[0:2, idx]
            prev_lidar_image[y, x] = prev_coloring_lidar[idx]

        # Remove points in vehicle bounding boxes
        for box in prev_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=prev_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                prev_lidar_image[min_y:max_y, min_x:max_x] = 0

        # Now we need to convert image format to point cloud array format
        prev_lidar_points_y, prev_lidar_points_x  = np.nonzero(prev_lidar_image)
        prev_lidar_points_z = prev_lidar_image[prev_lidar_points_y, prev_lidar_points_x]

        # Backproject to camera frame as 3 x N
        x_y_homogeneous = np.stack([
            prev_lidar_points_x,
            prev_lidar_points_y,
            np.ones_like(prev_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(prev_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(prev_lidar_points_z, axis=0)

        # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
        # So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)

        # Convert lidar point cloud into a nuScene LidarPointCloud object
        prev_point_cloud = LidarPointCloud(x_y_z)

        # Now we can transform the points back to the lidar frame of reference
        prev_point_cloud = camera_to_lidar_frame(
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=prev_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_lidar_main, prev_coloring_lidar_main, _ = point_cloud_to_image(
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        prev_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for idx in range(0, prev_points_lidar_main_quantized.shape[-1]):
            x, y = prev_points_lidar_main_quantized[0:2, idx]
            prev_lidar_image_main[y, x] = prev_coloring_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                prev_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0

        # Convert image format to point cloud format 2 x N
        prev_lidar_points_main_y, prev_lidar_points_main_x  = np.nonzero(prev_lidar_image_main)
        prev_lidar_points_main_z = prev_lidar_image_main[prev_lidar_points_main_y, prev_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        prev_points_lidar_main = np.stack([
            prev_lidar_points_main_x,
            prev_lidar_points_main_y],
            axis=0)
        prev_coloring_lidar_main = prev_lidar_points_main_z

        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for point_idx in range(0, prev_points_lidar_main_quantized.shape[1]):
            x = prev_points_lidar_main_quantized[0, point_idx]
            y = prev_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_coloring_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = prev_coloring_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = prev_coloring_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and coloring(depths). We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_lidar_y, return_points_lidar_x = np.nonzero(main_lidar_image)

    # Array of 1, N depth
    return_coloring_lidar = main_lidar_image[return_points_lidar_y, return_points_lidar_x]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_lidar = np.stack([
        return_points_lidar_x,
        return_points_lidar_y],
        axis=0)

    return return_points_lidar, return_coloring_lidar

def merge_radar_point_clouds2(nusc,
                              current_sample_token,
                              n_forward,
                              n_backward):
    """
    Merges Radar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
    Returns:
        numpy[float32] : 2 x N of x, y for radar points projected into the image
        numpy[float32] : N depths of radar points

    """

    nusc_explorer = NuScenesExplorer(nusc)

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_radar_token = current_sample['data']['RADAR_FRONT']

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    # Project the radar frame into the camera frame
    main_points_radar, main_coloring_radar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=main_radar_token,
        camera_token=main_camera_token)

    # Convert nuScenes format to numpy for image
    main_image = np.asarray(main_image)

    # Create an empty radar image
    main_radar_image = np.zeros((main_image.shape[0], main_image.shape[1]))

    main_points_radar_quantized = np.round(main_points_radar).astype(int)

    # Iterating through each radar point and plotting them onto the radar image
    for point_idx in range(0, main_points_radar_quantized.shape[1]):
        # Get x and y index in image frame
        x = main_points_radar_quantized[0, point_idx]
        y = main_points_radar_quantized[1, point_idx]

        # Value of y, x is the depth
        main_radar_image[y, x] = main_coloring_radar[point_idx]

    # Create a validity map to check which elements of the radar image are valid
    main_validity_map = np.where(main_radar_image > 0, 1, 0)

    # Count forward and backward frames
    n_forward_processed = 0
    n_backward_processed = 0

    # Initialize next sample as current sample
    next_sample = copy.deepcopy(current_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud in `next' frame,
        2. Project onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the next sample amd move forward
        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample', next_sample_token)

        # Get radar and camera token in the current sample
        next_radar_token = next_sample['data']['RADAR_FRONT']

        # Grab the radar sample
        next_radar_sample = nusc.get('sample_data', next_radar_token)

        # get the point cloud path and grab the radar point cloud
        next_radar_pcl_path = os.path.join(nusc.dataroot, next_radar_sample['filename'])
        RadarPointCloud.disable_filters()
        next_radar_point_cloud = RadarPointCloud.from_file(next_radar_pcl_path)
        # assert (rpc.points[2, :] == 0).all()

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_radar_main, next_coloring_radar_main, _ = point_cloud_to_image(
            point_cloud=next_radar_point_cloud,
            lidar_sensor_token=next_radar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        next_points_radar_main_quantized = np.round(next_points_radar_main).astype(int)

        for point_idx in range(0, next_points_radar_main_quantized.shape[1]):
            x = next_points_radar_main_quantized[0, point_idx]
            y = next_points_radar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_coloring_radar_main[point_idx] < main_radar_image[y, x]

            if is_not_occluded:
                main_radar_image[y, x] = next_coloring_radar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x] = next_coloring_radar_main[point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1

    # Initialize previous sample as current sample
    prev_sample = copy.deepcopy(current_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud in `prev' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the prev sample and move forward
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample', prev_sample_token)

        # Get radar and camera token in the current sample
        prev_radar_token = prev_sample['data']['RADAR_FRONT']

        # Grab the radar sample
        prev_radar_sample = nusc.get('sample_data', prev_radar_token)

        # get the point cloud path and grab the radar point cloud
        prev_radar_pcl_path = os.path.join(nusc.dataroot, prev_radar_sample['filename'])
        RadarPointCloud.disable_filters()
        prev_radar_point_cloud = RadarPointCloud.from_file(prev_radar_pcl_path)
        # assert (rpc.points[2, :] == 0).all()

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_radar_main, prev_coloring_radar_main, _ = point_cloud_to_image(
            point_cloud=prev_radar_point_cloud,
            lidar_sensor_token=prev_radar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        prev_points_radar_main_quantized = np.round(prev_points_radar_main).astype(int)

        for point_idx in range(0, prev_points_radar_main_quantized.shape[1]):
            x = prev_points_radar_main_quantized[0, point_idx]
            y = prev_points_radar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_coloring_radar_main[point_idx] < main_radar_image[y, x]

            if is_not_occluded:
                main_radar_image[y, x] = prev_coloring_radar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x] = prev_coloring_radar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and coloring(depths). We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_radar_y, return_points_radar_x = np.nonzero(main_radar_image)

    # Array of 1, N depth
    return_coloring_radar = main_radar_image[return_points_radar_y, return_points_radar_x]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_radar = np.stack([
        return_points_radar_x,
        return_points_radar_y],
        axis=0)

    return return_points_radar, return_coloring_radar

# Total scenes = 850
def point_registration(points_lidar, coloring_lidar, points_radar, coloring_radar, camera_token, z_scaling_factor=0.6):
    '''

    Arg(s):
        points_lidar :numpy[float32]
            x, y for lidar points array of size 2, N
        coloring_lidar : numpy[float32]
            z for lidar points array of size N
        points_radar : numpy[float32]
            x, y for radar points array of size 2, N
        coloring_radar : numpy[float32]
            z for radar points array of size N
        x_caling_factor: int
            divides the X values of lidar and radar points before feeding them into the KDTree
        z_caling_factor: int
            divides the Z values of lidar and radar points before feeding them into the KDTree
    Returns:
        numpy[float32] : lidar x, y coordinates N, 2
        numpy[float32] : radar x, y, z coordinates N, 3
        list[int] : index array to identify the ID of each point N
        list[numpy[float32]] : difference between lidar and radar list of x, y component shifts
    '''

    # Number of points for lidar and radar
    num_points_lidar = points_lidar.shape[1]
    num_points_radar = points_radar.shape[1]

    # We need to back-project the points from the camera coordinates to the lidar coordinates to conduct the search for nearest lidar points to radar

    # Get the camera intrinsics for backprojecting the points in lidar coordinates
    _, _, camera_intrinsics = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ANY,
        use_flat_vehicle_coordinates=False)

    # Backproject to camera frame as 3 x N
    x_y_homogeneous_lidar = np.stack([
        points_lidar[0, :],
        points_lidar[1, :],
        np.ones_like(points_lidar[0, :])],
        axis=0)

    x_y_lifted_lidar = np.matmul(np.linalg.inv(camera_intrinsics), x_y_homogeneous_lidar)
    x_y_z_lidar = x_y_lifted_lidar * np.expand_dims(coloring_lidar, axis=0)

    # Backproject to camera frame as 3 x N
    x_y_homogeneous_radar = np.stack([
        points_radar[0, :],
        points_radar[1, :],
        np.ones_like(points_radar[0, :])],
        axis=0)

    x_y_lifted_radar = np.matmul(np.linalg.inv(camera_intrinsics), x_y_homogeneous_radar)
    x_y_z_radar = x_y_lifted_radar * np.expand_dims(coloring_radar, axis=0)

    # Construct training set of KDTree in form N, 2 for (x, z)
    X_lidar = np.empty([num_points_lidar, 2])

    # Set x component as feature
    X_lidar[:, 0] = x_y_z_lidar[0, :]

    # Set z component as feature and weigh it 100x more than x component
    X_lidar[:, 1] = coloring_lidar * z_scaling_factor

    # Construct query set for KDTree in form N, 2 for (x, z)
    X_radar = np.empty([num_points_radar, 2])

    # Set x component as feature
    X_radar[:, 0] = x_y_z_radar[0, :]

    # Set z component as feature and weigh it 100x more than x component
    X_radar[:, 1] = coloring_radar * z_scaling_factor

    # Train KDTree
    tree = KDTree(X_lidar)

    # Create list to store the values of correspondece points in lidar and radar
    lidar_corresponding_point_x = []
    lidar_corresponding_point_y = []
    lidar_corresponding_point_z = []

    radar_corresponding_point_x = []
    radar_corresponding_point_y = []
    radar_corresponding_point_z = []

    # Create an empty list ot store the index of the points - tells how mnay pointd are there and is used in later functions
    idx_array = []

    # Iterate through radar points in X_radar and query the tree
    for radar_point_index in range(0, X_radar.shape[0]):
        # Input to the query_radius function is of the shape (n_samples, n_features)
        radar_point_query = np.expand_dims(X_radar[radar_point_index], 0)

        # Query for points in radar point cloud that are within 25 cm of the lidar points
        index, distance = tree.query_radius(
            radar_point_query,
            r=KDTREE_QUERY_RADIUS,
            count_only=False,
            return_distance=True,
            sort_results=True)

        # index is of the form (index, distance). So we get the indices
        indices = index[0]

        # If no correspondence is found between the lidar and radar points, just add a -1 to the lidar which is taken care of in validity map later
        if len(indices) == 0:
            '''
            lidar_corresponding_point_x.append(-1)
            lidar_corresponding_point_y.append(-1)
            lidar_corresponding_point_z.append(-1)
            radar_corresponding_point_x.append(points_radar[0, radar_point_index])
            radar_corresponding_point_y.append(points_radar[1, radar_point_index])
            radar_corresponding_point_z.append(coloring_radar[radar_point_index])
            idx_array.append(radar_point_index)
            '''
            continue
        else:
            for corresponding_point_index in range(0, len(indices)):
                lidar_corresponding_point_x.append(points_lidar[0, indices[corresponding_point_index]])
                lidar_corresponding_point_y.append(points_lidar[1, indices[corresponding_point_index]])
                lidar_corresponding_point_z.append(coloring_lidar[indices[corresponding_point_index]])
                radar_corresponding_point_x.append(points_radar[0, radar_point_index])
                radar_corresponding_point_y.append(points_radar[1, radar_point_index])
                radar_corresponding_point_z.append(coloring_radar[radar_point_index])
                idx_array.append(radar_point_index)

    # Original radar points with garbage y cooordinates
    corresponding_radar_points = np.stack([
        radar_corresponding_point_x,
        radar_corresponding_point_y,
        radar_corresponding_point_z],
        axis=-1)

    # These points will serve as out ground truth while training
    corresponding_lidar_points = np.stack([
        lidar_corresponding_point_x,
        lidar_corresponding_point_y,
        lidar_corresponding_point_z],
        axis=-1)

    return corresponding_lidar_points, corresponding_radar_points, idx_array


def check_for_length_and_copy(input_array, final_len):
    '''
    Takes a list as input and copies the first element again and again till the length of the list becomes equal to final_len

    Arg(s):
        input_array : list
        final_len : final length that you want the list to be of

    Returns:
        numpy[] : array of length = final_len
    '''
    # If we have more points than the final length we want, we need to truncate the array. This is used when a radar point has more than final_len close lidar_points
    if len(input_array) > final_len:
        input_array = input_array[:final_len]

    elif len(input_array) < final_len:
        while len(input_array) < final_len:
            input_array.append(input_array[0])

    # convert the list to numpy array
    input_array = np.asarray(input_array)
    return np.asarray(input_array)

def reorganize_points(radar_points, lidar_points, idx_array, num_points):
    '''
    Arg(s):
        radar_points : radar points in the form x, y, z (3, N)
        lidar_points : lidar points in the x, y, z (3, N)
        idx_array : contains index of radar points in the sample
        num_points : number of points that we want in our ground truth closest to the radar point

    Returns:
        numpy[float32] : total_radar_points_in_sample, num_points, 3
        numpy[float32] : total_radar_points_in_sample, num_points, 2

    '''
    radar_list = []
    lidar_list = []

    # Final outputs
    radar_points_set = []
    lidar_points_set = []

    # Flag checks which index we have iterated over
    flag = idx_array[0]

    # for every point in the same sample, we add it to a list and then copy the list till we reach the goal number of points. In some cases we may have more than the number of points we want, so we truncate
    for i in range(0, len(idx_array)):
        if idx_array[i] == flag:
            # First we take each point in the form x, y, z
            radar_point_x_y_z = [radar_points[i][0], radar_points[i][1], radar_points[i][2]]
            lidar_point_x_y_z = [lidar_points[i][0], lidar_points[i][1]]

            # Add each point to their respective lists
            radar_list.append(radar_point_x_y_z)
            lidar_list.append(lidar_point_x_y_z)

        elif idx_array[i] != flag:
            # now we have moved to the next sample so we will store the current points into the final output
            flag = idx_array[i]
            # make the lists into numpy arrays of the specified size
            radar_list = check_for_length_and_copy(radar_list, num_points)
            lidar_list = check_for_length_and_copy(lidar_list, num_points)

            # Check if radar_list_big is empty
            if len(radar_points_set) == 0:
                radar_points_set = radar_list
                lidar_points_set = lidar_list
            # We want to return in the form total_points, num_points, 2/3
            else:
                radar_list = np.expand_dims(radar_list, axis=0)
                lidar_list = np.expand_dims(lidar_list, axis=0)
                # We want to return in the form total_points, num_points, 2/3
                if radar_points_set.shape == (num_points, 3):
                    radar_points_set = np.expand_dims(radar_points_set, axis=0)
                    lidar_points_set = np.expand_dims(lidar_points_set, axis=0)
                # Add the points to the final set
                radar_points_set = np.concatenate((radar_points_set, radar_list), axis=0)
                lidar_points_set = np.concatenate((lidar_points_set, lidar_list), axis=0)

            # Do the same thing for the points in the new sample
            radar_list = []
            lidar_list = []
            radar_point_x_y_z = [radar_points[i][0], radar_points[i][1], radar_points[i][2]]
            lidar_point_x_y_z = [lidar_points[i][0], lidar_points[i][1]]
            radar_list.append(radar_point_x_y_z)
            lidar_list.append(lidar_point_x_y_z)

    # we will have the last set of points not accounted for in the final output so we store them
    if len(radar_points_set) == 0:
        radar_list = check_for_length_and_copy(radar_list, num_points)
        lidar_list = check_for_length_and_copy(lidar_list, num_points)
        radar_points_set = radar_list
        lidar_points_set = lidar_list
    else:
        radar_list = check_for_length_and_copy(radar_list, num_points)
        lidar_list = check_for_length_and_copy(lidar_list, num_points)
        radar_list = np.expand_dims(radar_list, axis=0)
        lidar_list = np.expand_dims(lidar_list, axis=0)

        if radar_points_set.shape == (num_points, 3):
            radar_points_set = np.expand_dims(radar_points_set, axis=0)
            lidar_points_set = np.expand_dims(lidar_points_set, axis=0)

        radar_points_set = np.concatenate((radar_points_set, radar_list), axis=0)
        lidar_points_set = np.concatenate((lidar_points_set, lidar_list), axis=0)
    return radar_points_set, lidar_points_set

def lidar_point_clouds_return_depth_map(nusc, current_sample_token):
    """
    Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.
    :param nusc: NuScenes Object.
    :param current_sample_token: (str) token for the current sample.
    :return: <np.ndarray: image, np.ndarray: validity_map>.
    """

    nusc_explorer = NuScenesExplorer(nusc)
    my_sample = nusc.get('sample', current_sample_token)
    lidar_token = my_sample['data']['LIDAR_TOP']
    main_camera_token = my_sample['data']['CAM_FRONT']

    # project the lidar frame into the camera frame
    main_points_lidar, main_coloring_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_coloring_lidar, main_image)

    return depth_map

def points_to_depth_map(points, coloring, image):
    '''
    Plots the depth values onto the image plane

    Arg(s):
        points : numpy[float32]
            2 x N matrix in x, y
        coloring : numpy[float32]
            N scales for z
        image : numpy[float32]
            H x W x 3 image for reference frame size
    Returns:
        numpy[float32] : H x W image with depth plotted
    '''

    # Plot points onto the image
    image = np.asarray(image)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    points_quantized = np.round(points).astype(int)

    for pt_idx in range(0, points_quantized.shape[1]):
        x = points_quantized[0, pt_idx]
        y = points_quantized[1, pt_idx]
        depth_map[y, x] = coloring[pt_idx]

    return depth_map

def process_scene(args):
    '''
    Processes one scene from first sample to last sample

    Arg(s):
        args : tuple(int, str, str)
            scene_id : int
                identifier for one scene
            first_sample_token : str
                token to identify first sample in the scene for fetching
            last_sample_token : str
                token to identify last sample in the scene for fetching
    Returns:
        dict[(int, int), Data_Struct] : dictionary mapping from (scene id, sample id) to a data structure
    '''

    scene_id, first_sample_token, last_sample_token = args

    # Output dictionary of sample id to data structures
    data_dict = dict()

    # Instantiate the first sample id
    sample_id = 0
    sample_token = first_sample_token

    print('Processing scene_id={}'.format(scene_id))

    # Iterate through all samples up to the last sample
    while sample_token != last_sample_token:

        # Fetch a single sample
        my_sample = nusc.get('sample', sample_token)
        camera_token = my_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        # Get Camera Data
        camera_image = image.imread(os.path.join(nusc.dataroot, camera_sample['filename']))

        # Transform Lidar and Radar Points to the image coordinate
        points_radar, coloring_radar = merge_radar_point_clouds2(
            nusc,
            sample_token,
            n_forward=n_forward,
            n_backward=n_backward)

        # Merges n_forward and n_backward number of point clouds to frame at sample token
        points_lidar, coloring_lidar = merge_lidar_point_clouds2(
            nusc,
            sample_token,
            n_forward=n_forward,
            n_backward=n_backward)

        # ================== Begin Lidar image save code =============================
        # Merges n_forward and n_backward number of point clouds to frame at sample token but in this case we need the lidar image so that we can save it
        ground_truth_depth = points_to_depth_map(points_lidar, coloring_lidar, camera_image)

        lidar_image_dir_path, lidar_image_file_name = os.path.split(camera_sample['filename'])
        lidar_image_dir_path = lidar_image_dir_path.replace(
            'samples',
            os.path.join('data', 'ground_truth_train-scene-id{}'.format(scene_id)))
        lidar_image_file_name = os.path.splitext(lidar_image_file_name)[0] + '.png'

        ground_truth_depth_output_path = os.path.join(lidar_image_dir_path, lidar_image_file_name)

        # In case multiple threads create same directory
        if not os.path.exists(lidar_image_dir_path):
            try:
                os.makedirs(lidar_image_dir_path)
            except Exception:
                pass

        save_depth(ground_truth_depth, ground_truth_depth_output_path)
        # ========= End lidar image save code ==============

        # Perform registration: maps radar point to ground truth lidar points
        ground_truth_points, input_points, idx_array = point_registration(
            points_lidar,
            coloring_lidar,
            points_radar,
            coloring_radar,
            camera_token,
            KDTREE_DEPTH_WEIGHT)

        if len(idx_array) == 0:
            print('Found empty scene, skipping scene: scene_id={}'.format(scene_id))
            break

        # Reorganize points outputs num_radar_points, num_points, 2/3 format
        input_points, ground_truth_points = reorganize_points(
            input_points,
            ground_truth_points,
            idx_array,
            num_points)

        # Some sample may only 1 radar point, which drops a dimension to (N, 2)
        # where N (num_points) number of ground truth points
        if ground_truth_points.ndim == 2:
            # We add it back so it is (1, N, 2) for one radar point
            ground_truth_points = np.expand_dims(ground_truth_points, 0)

        # Use 2 to denote class of all invalid points
        lidar_label = np.full(camera_image.shape[0:2], 2)

        # Iterate through N lidar points
        points_lidar_quantized = np.round(points_lidar).astype(int)
        for idx in range(0, points_lidar.shape[-1]):
            # Fetch x, y coordinate and quantize to pixel coordinate
            x, y = points_lidar_quantized[0:2, idx]
            lidar_label[y, x] = 0

        # Make lidar frame H x W x 1 and repeat the frame N times
        lidar_labels = np.expand_dims(lidar_label, -1)
        lidar_labels = np.tile(lidar_labels, (1, 1, ground_truth_points.shape[0]))

        # For every ground truth lidar point and radar point
        ground_truth_points_quantized = np.round(ground_truth_points).astype(int)

        for idx_channel in range(0, ground_truth_points_quantized.shape[0]):
            for idx_point in range(0, ground_truth_points_quantized.shape[1]):
                # We mark all ground truth matching points based on closness to 1
                x, y = ground_truth_points_quantized[idx_channel, idx_point, :]

                if x > 0 and y > 0:
                    lidar_labels[y, x, idx_channel] = 1

        # Store ground truth label (0, 1, 2), 0 too far, 1 close, 2 invalid
        lidar_labels = np.uint8(lidar_labels)

        dir_path, file_name = os.path.split(camera_sample['filename'])
        dir_path = dir_path.replace(
            'samples',
            os.path.join('data', 'pseudo_ground_truth_train-scene-id{}'))
        dir_path = dir_path.format(scene_id)
        file_name = os.path.splitext(file_name)[0] + '-{}.png'

        label_output_path = os.path.join(dir_path, file_name)

        # In case multiple threads create same directory
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception:
                pass

        # Save each ground truth label map as a PNG file
        for idx_channel in range(0, ground_truth_points.shape[0]):
            ground_truth_label_output_path = label_output_path.format(idx_channel)
            output_png = Image.fromarray(np.squeeze(lidar_labels[:, :, idx_channel]), mode='L')
            output_png.save(ground_truth_label_output_path)

        dir_path, file_name = os.path.split(camera_sample['filename'])
        dir_path = dir_path.replace(
            'samples',
            os.path.join('data', 'radar_points-scene-id{}'))
        dir_path = dir_path.format(scene_id)
        file_name = os.path.splitext(file_name)[0] + '-{}.npy'.format(sample_id)

        # In case multiple threads create same directory
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception:
                pass

        radar_points_output_path = os.path.join(dir_path, file_name)

        np.save(radar_points_output_path, input_points)

        data_item = Data_Struct(
            scene_id,
            sample_id,
            camera_sample['filename'],
            radar_points_output_path,
            ground_truth_label_output_path,
            ground_truth_depth_output_path)

        data_dict[(scene_id, sample_id)] = []
        data_dict[(scene_id, sample_id)].append(data_item)
        sample_id = sample_id + 1

        sample_token = my_sample['next']

    print('Finished {} samples in scene_id={}'.format(sample_id, scene_id))

    return data_dict


'''
Main function
'''
data_dict = dict()
dict_idx = 0

DEBUG = False

pool_inputs = []

train_ids, val_ids = get_train_val_split_ids()

# Add all tasks for processing each scene to pool inputs
for scene_id in range(0, max_scenes):
    if scene_id not in train_ids:
        continue
    my_scene = nusc.scene[scene_id]
    first_sample_token = my_scene['first_sample_token']
    last_sample_token = my_scene['last_sample_token']

    pool_inputs.append((scene_id, first_sample_token, last_sample_token))

    if DEBUG:
        result = process_scene((scene_id, first_sample_token, last_sample_token))

        for key in result.keys():
            data_dict[dict_idx] = result[key]
            dict_idx = dict_idx + 1

if not DEBUG:
    # Create pool of threads
    with mp.Pool(n_thread) as pool:
        # Will fork n_thread to process scene
        pool_results = pool.map(process_scene, pool_inputs)

        # Returns a list of dictionaries containing scene info
        for result in pool_results:

            # Iterate through each key
            for key in result.keys():
                data_dict[dict_idx] = result[key]
                dict_idx = dict_idx + 1

f = open(pickle_file_name, "wb")
pickle.dump(data_dict, f)
f.close()
