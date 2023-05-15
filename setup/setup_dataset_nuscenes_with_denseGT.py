from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, copy, argparse
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import multiprocessing as mp

sys.path.insert(0, 'src')
import data_utils

MAX_SCENES = 850


'''
Output filepaths
'''
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')

TRAIN_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_image.txt')
TRAIN_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_lidar.txt')
TRAIN_RADAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_radar.txt')
TRAIN_RADAR_REPROJECTED_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_radar_reprojected.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_ground_truth.txt')
TRAIN_GROUND_TRUTH_INTERP_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_ground_truth_interp.txt')

VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image.txt')
VAL_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar.txt')
VAL_RADAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_radar.txt')
VAL_RADAR_REPROJECTED_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_radar_reprojected.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth.txt')
VAL_GROUND_TRUTH_INTERP_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth_interp.txt')

VAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image-subset.txt')
VAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar-subset.txt')
VAL_RADAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_radar-subset.txt')
VAL_RADAR_REPROJECTED_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_radar_reprojected-subset.txt')
VAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth-subset.txt')
VAL_GROUND_TRUTH_INTERP_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth_interp-subset.txt')

'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_scenes_to_process',
    type=int, default=MAX_SCENES, help='Number of scenes to process')
parser.add_argument('--n_forward_frames_to_reproject',
    type=int, default=12, help='Number of forward frames to project onto a target frame')
parser.add_argument('--n_backward_frames_to_reproject',
    type=int, default=12, help='Number of backward frames to project onto a target frame')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=40, help='Number of threads to use in parallel pool')
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')
parser.add_argument('--panoptic_seg_dir',
    type=str, required=True, help='path to panoptic segmentation masks')


args = parser.parse_args()


# Create global nuScene object
nusc = NuScenes(
    version='v1.0-trainval',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer = NuScenesExplorer(nusc)

def get_train_val_split_ids(debug=False):
    '''
    Given the nuscenes object, find out which scene ids correspond to which set.
    The split is taken from the official nuScene split available here:
    https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

    Arg(s):
        debug : bool
            if set, then enter debug mode

    Returns:
        list[int] : list containing ids of the scenes that are training split
        list[int] : list containing ids of the scenes that are validation split
    '''

    train_file_name = os.path.join('data_split', 'train_ids.pkl')
    val_file_name = os.path.join('data_split', 'val_ids.pkl')

    open_file = open(train_file_name, "rb")
    train_ids = pickle.load(open_file)
    open_file.close()

    open_file = open(val_file_name, "rb")
    val_ids = pickle.load(open_file)
    open_file.close()

    if debug:
        train_ids_final = [1]
        return train_ids_final, val_ids

    return train_ids, val_ids

def point_cloud_to_image(nusc,
                         point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
        minimum_distance_from_camera : float32
            threshold for removing points that exceeds minimum distance from camera
    Returns:
        numpy[float32] : 3 x N array of x, y, z
        numpy[float32] : N array of z
        numpy[float32] : camera image
    '''

    # Get dictionary of containing path to image, pose, etc.
    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image_path = os.path.join(nusc.dataroot, camera['filename'])
    image = data_utils.load_image(image_path)

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
    depth = point_cloud.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Points will be 3 x N
    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

    # Select points that are more than min distance from camera and not on edge of image
    points = points[:, mask]
    depth = depth[mask]

    return points, depth, image

def camera_to_lidar_frame(nusc,
                          point_cloud,
                          lidar_sensor_token,
                          camera_token):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
    Returns:
        PointCloud : nuScenes point cloud object
    '''

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

def merge_lidar_point_clouds(nusc,
                             nusc_explorer,
                             current_sample_token,
                             n_forward,
                             n_backward,
                             lidar_camera_correspondence_dict,
                             panoptic_seg_dir):
    '''
    Merges Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
        lidar_camera_correspondence_dict : dict
            contains closest camera token to every lidar token in the scene
        panoptic_seg_dir : str
            directory where all the panoptic segmentation masks are stored
    Returns:
        numpy[float32] : 2 x N of x, y for lidar points projected into the image
        numpy[float32] : N depths of lidar points

    '''

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_lidar_token = current_sample['data']['LIDAR_TOP']

    # Get lidar data
    main_lidar_sample= nusc.get('sample_data', main_lidar_token)

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    # load the panoptic segmentation mask for the main camera token
    main_camera_panoptic_mask_path = os.path.join(panoptic_seg_dir, main_camera_token + '.npy')
    main_camera_panoptic_mask = np.load(main_camera_panoptic_mask_path)

    # Project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
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
        main_lidar_image[y, x] = main_depth_lidar[point_idx]

    # Create a validity map to check which elements of the lidar image are valid
    main_validity_map = np.where(main_lidar_image > 0, 1, 0)

    # Count forward and backward frames
    n_forward_processed = 0
    n_backward_processed = 0

    # Initialize next sample as current sample. If we choose the current_sample, we can only go to other key frames. In order to go to intermediate frames, we need lidar sample
    next_sample = copy.deepcopy(main_lidar_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud in `next' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the next sample amd move forward
        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample_data', next_sample_token)

        # Get camera token closest to the current lidar token
        closest_camera_token = lidar_camera_correspondence_dict[next_sample_token]

        # Get bounding box in image frame to remove vehicles from point cloud
        _, next_boxes, next_camera_intrinsics = nusc.get_sample_data(
            closest_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map next frame point cloud to image so we can remove vehicle based on panoptic mask
        next_points_lidar, next_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=next_sample_token,
            camera_token=closest_camera_token)

        next_lidar_image = np.zeros_like(main_lidar_image)

        next_points_lidar_quantized = np.round(next_points_lidar).astype(int)
        # Plots depth values onto the image
        for idx in range(0, next_points_lidar_quantized.shape[-1]):
            x, y = next_points_lidar_quantized[0:2, idx]
            next_lidar_image[y, x] = next_depth_lidar[idx]

        # load camera mask -- mask out all moving objects
        camera_panoptic_mask_path = os.path.join(panoptic_seg_dir, closest_camera_token + '.npy')
        camera_panoptic_mask = np.load(camera_panoptic_mask_path)

        # mask out all moving objects
        next_lidar_image[np.where(camera_panoptic_mask)] = 0

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
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_sample_token,
            camera_token=closest_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_lidar_main, next_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_sample_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        next_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for idx in range(0, next_points_lidar_main_quantized.shape[-1]):
            x, y = next_points_lidar_main_quantized[0:2, idx]
            next_lidar_image_main[y, x] = next_depth_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        next_lidar_image_main[np.where(main_camera_panoptic_mask)] = 0

        # Convert image format to point cloud format
        next_lidar_points_main_y, next_lidar_points_main_x  = np.nonzero(next_lidar_image_main)
        next_lidar_points_main_z = next_lidar_image_main[next_lidar_points_main_y, next_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        next_points_lidar_main = np.stack([
            next_lidar_points_main_x,
            next_lidar_points_main_y],
            axis=0)
        next_depth_lidar_main = next_lidar_points_main_z

        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for point_idx in range(0, next_points_lidar_main_quantized.shape[1]):
            x = next_points_lidar_main_quantized[0, point_idx]
            y = next_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1

    # Initialize previous sample as current sample
    prev_sample = copy.deepcopy(main_lidar_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud in `prev' frame,
        2. Poject onto image to remove vehicle bounding boxes
        3. Backproject to camera frame
        '''

        # Get the token and sample data for the previous sample and move sample backward
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample_data', prev_sample_token)

        # Get camera token closest to the current lidar token
        closest_camera_token = lidar_camera_correspondence_dict[prev_sample_token]

        # Get bounding box in image frame to remove vehicles from point cloud
        _, prev_boxes, prev_camera_intrinsics = nusc.get_sample_data(
            closest_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map next frame point cloud to image so we can remove vehicle based on panoptic mask
        prev_points_lidar, prev_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=prev_sample_token,
            camera_token=closest_camera_token)

        prev_lidar_image = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_quantized = np.round(prev_points_lidar).astype(int)

        for idx in range(0, prev_points_lidar_quantized.shape[-1]):
            x, y = prev_points_lidar_quantized[0:2, idx]
            prev_lidar_image[y, x] = prev_depth_lidar[idx]

        # load camera mask -- mask out all moving objects
        camera_panoptic_mask_path = os.path.join(panoptic_seg_dir, closest_camera_token + '.npy')
        camera_panoptic_mask = np.load(camera_panoptic_mask_path)

        # mask out all moving objects
        prev_lidar_image[np.where(camera_panoptic_mask)] = 0

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
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_sample_token,
            camera_token=closest_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_lidar_main, prev_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_sample_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        prev_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for idx in range(0, prev_points_lidar_main_quantized.shape[-1]):
            x, y = prev_points_lidar_main_quantized[0:2, idx]
            prev_lidar_image_main[y, x] = prev_depth_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        prev_lidar_image_main[np.where(main_camera_panoptic_mask)] = 0

        # Convert image format to point cloud format 2 x N
        prev_lidar_points_main_y, prev_lidar_points_main_x  = np.nonzero(prev_lidar_image_main)
        prev_lidar_points_main_z = prev_lidar_image_main[prev_lidar_points_main_y, prev_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        prev_points_lidar_main = np.stack([
            prev_lidar_points_main_x,
            prev_lidar_points_main_y],
            axis=0)
        prev_depth_lidar_main = prev_lidar_points_main_z

        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for point_idx in range(0, prev_points_lidar_main_quantized.shape[1]):
            x = prev_points_lidar_main_quantized[0, point_idx]
            y = prev_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and depth. We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_lidar_y, return_points_lidar_x = np.nonzero(main_lidar_image)

    # Array of 1, N depth
    return_depth_lidar = main_lidar_image[return_points_lidar_y, return_points_lidar_x]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_lidar = np.stack([
        return_points_lidar_x,
        return_points_lidar_y],
        axis=0)

    return return_points_lidar, return_depth_lidar

def merge_radar_point_clouds(nusc,
                             nusc_explorer,
                             current_sample_token,
                             n_forward,
                             n_backward):
    '''
    Merges Radar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
    Returns:
        numpy[float32] : 2 x N of x, y for radar points projected into the image
        numpy[float32] : N depths of radar points

    '''

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_radar_token = current_sample['data']['RADAR_FRONT']

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    # Project the radar frame into the camera frame
    main_points_radar, main_depth_radar, main_image = nusc_explorer.map_pointcloud_to_image(
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
        main_radar_image[y, x] = main_depth_radar[point_idx]

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

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_radar_main, next_depth_radar_main, _ = point_cloud_to_image(
            nusc=nusc,
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
                next_depth_radar_main[point_idx] < main_radar_image[y, x]

            if is_not_occluded:
                main_radar_image[y, x] = next_depth_radar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x] = next_depth_radar_main[point_idx]
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

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_radar_main, prev_depth_radar_main, _ = point_cloud_to_image(
            nusc=nusc,
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
                prev_depth_radar_main[point_idx] < main_radar_image[y, x]

            if is_not_occluded:
                main_radar_image[y, x] = prev_depth_radar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_radar_image[y, x] = prev_depth_radar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and depth. We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_radar_y, return_points_radar_x = np.nonzero(main_radar_image)

    # Array of 1, N depth
    return_depth_radar = main_radar_image[return_points_radar_y, return_points_radar_x]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_radar = np.stack([
        return_points_radar_x,
        return_points_radar_y],
        axis=0)

    return return_points_radar, return_depth_radar

def lidar_depth_map_from_token(nusc,
                               nusc_explorer,
                               current_sample_token):
    '''
    Picks current_sample_token as reference and projects lidar points onto the image plane.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
    Returns:
        numpy[float32] : H x W depth
    '''

    current_sample = nusc.get('sample', current_sample_token)
    lidar_token = current_sample['data']['LIDAR_TOP']
    main_camera_token = current_sample['data']['CAM_FRONT']

    # project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_depth_lidar, main_image)

    return depth_map

def points_to_depth_map(points, depth, image):
    '''
    Plots the depth values onto the image plane

    Arg(s):
        points : numpy[float32]
            2 x N matrix in x, y
        depth : numpy[float32]
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
        depth_map[y, x] = depth[pt_idx]

    return depth_map

def process_scene(args):
    '''
    Processes one scene from first sample to last sample

    Arg(s):
        args : tuple(Object, Object, str, int, str, str, int, int, str, bool)
            nusc : NuScenes Object
                nuScenes object instance
            nusc_explorer : NuScenesExplorer Object
                nuScenes explorer object instance
            tag : str
                train, val
            scene_id : int
                identifier for one scene
            lidar_camera_correspondence_dict : dict
                contains correspondence between a lidar token and its closest camera token
            panoptic_seg_dir : str
                directory where all the panoptic segmentation masks are stored
            first_sample_token : str
                token to identify first sample in the scene for fetching
            last_sample_token : str
                token to identify last sample in the scene for fetching
            n_forward : int
                number of forward (future) frames to reproject
            n_backward : int
                number of backward (previous) frames to reproject
            output_dirpath : str
                root of output directory
            paths_only : bool
                if set, then only produce paths
    Returns:
        list[str] : paths to camera image
        list[str] : paths to lidar depth map
        list[str] : paths to radar depth map
        list[str] : paths to ground truth (merged lidar) depth map
        list[str] : paths to ground truth (merged lidar) interpolated depth map
    '''

    tag, \
        scene_id, \
        lidar_camera_correspondence_dict, \
        panoptic_seg_dir, \
        first_sample_token, \
        last_sample_token, \
        n_forward, \
        n_backward, \
        output_dirpath, \
        paths_only = args

    # Instantiate the first sample id
    sample_id = 0
    sample_token = first_sample_token

    camera_image_paths = []
    lidar_paths = []
    radar_points_paths = []
    radar_points_reprojected_paths = []
    ground_truth_paths = []
    ground_truth_interp_paths = []

    print('Processing scene_id={}'.format(scene_id))

    # Iterate through all samples up to the last sample
    while sample_token != last_sample_token:

        # Fetch a single sample
        current_sample = nusc.get('sample', sample_token)
        camera_token = current_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        '''
        Set up paths
        '''
        camera_image_path = os.path.join(nusc.dataroot, camera_sample['filename'])

        dirpath, filename = os.path.split(camera_image_path)
        dirpath = dirpath.replace(nusc.dataroot, output_dirpath)
        filename = os.path.splitext(filename)[0]

        # Create lidar path
        lidar_dirpath = dirpath.replace(
            'samples',
            os.path.join('lidar', 'scene_{}'.format(scene_id)))
        lidar_filename = filename + '.png'

        lidar_path = os.path.join(
            lidar_dirpath,
            lidar_filename)

        # Create radar path
        radar_points_dirpath = dirpath.replace(
            'samples',
            os.path.join('radar_points', 'scene_{}'.format(scene_id)))
        radar_points_filename = filename + '.npy'

        radar_points_path = os.path.join(
            radar_points_dirpath,
            radar_points_filename)

        radar_points_reprojected_dirpath = dirpath.replace(
            'samples',
            os.path.join('radar_points_reprojected', 'scene_{}'.format(scene_id)))

        radar_points_reprojected_path = os.path.join(
            radar_points_reprojected_dirpath,
            radar_points_filename)

        # Create ground truth path
        ground_truth_dirpath = dirpath.replace(
            'samples',
            os.path.join('ground_truth', 'scene_{}'.format(scene_id)))
        ground_truth_filename = filename + '.png'

        ground_truth_path = os.path.join(
            ground_truth_dirpath,
            ground_truth_filename)

        # Create interpolated densified ground truth path
        ground_truth_interp_dirpath = dirpath.replace(
            'samples',
            os.path.join('ground_truth_interp', 'scene_{}'.format(scene_id)))
        ground_truth_interp_filename = filename + '.png'

        ground_truth_interp_path = os.path.join(
            ground_truth_interp_dirpath,
            ground_truth_interp_filename)

        # In case multiple threads create same directory
        dirpaths = [
            lidar_dirpath,
            radar_points_dirpath,
            radar_points_reprojected_dirpath,
            ground_truth_dirpath,
            ground_truth_interp_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                except Exception:
                    pass

        '''
        Store file paths
        '''
        camera_image_paths.append(camera_image_path)
        radar_points_paths.append(radar_points_path)
        radar_points_reprojected_paths.append(radar_points_reprojected_path)
        lidar_paths.append(lidar_path)
        ground_truth_paths.append(ground_truth_path)
        ground_truth_interp_paths.append(ground_truth_interp_path)

        if not paths_only:

            '''
            Get camera data
            '''
            camera_image = data_utils.load_image(camera_image_path)

            '''
            Get lidar points projected to an image and save as PNG
            '''
            lidar_depth = lidar_depth_map_from_token(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token)

            data_utils.save_depth(lidar_depth, lidar_path)

            '''
            Merge forward and backward point clouds for radar and lidar
            '''
            # Transform Lidar and Radar Points to the image coordinate
            points_radar_reprojected, depth_radar_reprojected = merge_radar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=n_forward,
                n_backward=n_backward)

            points_radar, depth_radar = merge_radar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=0,
                n_backward=0)

            # Merges n_forward and n_backward number of point clouds to frame at sample token
            points_lidar, depth_lidar = merge_lidar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=n_forward,
                n_backward=n_backward,
                lidar_camera_correspondence_dict=lidar_camera_correspondence_dict,
                panoptic_seg_dir=panoptic_seg_dir)

            '''
            Project point cloud onto the image plane and save as PNG
            '''
            # Merges n_forward and n_backward number of point clouds to frame at sample token
            # but in this case we need the lidar image so that we can save it
            ground_truth = points_to_depth_map(points_lidar, depth_lidar, camera_image)

            # Save depth map as PNG
            data_utils.save_depth(ground_truth, ground_truth_path)

            '''
            Interpolate missing points in ground truth point cloud and save as PNG
            '''
            validity_map = np.where(ground_truth > 0.0, 1.0, 0.0)
            ground_truth_interp = data_utils.interpolate_depth(
                ground_truth,
                validity_map)

            # Save depth map as PNG
            data_utils.save_depth(ground_truth_interp, ground_truth_interp_path)

            '''
            Save radar points as a numpy array
            '''
            radar_points_reprojected = np.stack([
                points_radar_reprojected[0, :],
                points_radar_reprojected[1, :],
                depth_radar_reprojected],
                axis=-1)

            radar_points = np.stack([
                points_radar[0, :],
                points_radar[1, :],
                depth_radar],
                axis=-1)

            np.save(radar_points_reprojected_path, radar_points_reprojected)
            np.save(radar_points_path, radar_points)

        '''
        Move to next sample in scene
        '''
        sample_id = sample_id + 1
        sample_token = current_sample['next']

    print('Finished {} samples in scene_id={}'.format(sample_id, scene_id))

    return (tag,
            camera_image_paths,
            lidar_paths,
            radar_points_paths,
            radar_points_reprojected_paths,
            ground_truth_paths,
            ground_truth_interp_paths)


def get_closest_camera_image_to_lidar_scan_in_a_sample(scene):
    '''
    For a scene, returns a dictionary specifying closest camera token to a lidar token in that scene
    '''
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    lidar_token = first_sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_time_stamps_dict = dict()
    camera_time_stamps_dict = dict()
    camera_time_stamps = []
    while lidar_data['next'] != '':
        lidar_time_stamps_dict[lidar_token] = lidar_data['timestamp']
        lidar_token = lidar_data['next']                    
        lidar_data = nusc.get('sample_data', lidar_token)
    lidar_time_stamps_dict[lidar_token] = lidar_data['timestamp']
    
    camera_token = first_sample['data']['CAM_FRONT']
    camera_data = nusc.get('sample_data', camera_token)
    while camera_data['next'] != '':
        camera_time_stamps_dict[camera_data['timestamp']] = camera_token
        camera_time_stamps.append(camera_data['timestamp'])
        camera_token = camera_data['next']                    
        camera_data = nusc.get('sample_data', camera_token)
    camera_time_stamps_dict[camera_data['timestamp']] = camera_token
    camera_time_stamps.append(camera_data['timestamp'])
    
    lidar_camera_correspondence_dict = dict()
    for lidar_token in lidar_time_stamps_dict:
        camera_idx = min(camera_time_stamps, key=lambda x:abs(x-lidar_time_stamps_dict[lidar_token]))
        lidar_camera_correspondence_dict[lidar_token] = camera_time_stamps_dict[camera_idx]
        
    return lidar_camera_correspondence_dict


'''
Main function
'''
if __name__ == '__main__':

    use_multithread = args.n_thread > 1 and not args.debug

    pool_inputs = []
    pool_results = []

    train_camera_image_paths = []
    train_lidar_paths = []
    train_radar_points_paths = []
    train_radar_points_reprojected_paths = []
    train_ground_truth_paths = []
    train_ground_truth_interp_paths = []

    val_camera_image_paths = []
    val_lidar_paths = []
    val_radar_points_paths = []
    val_radar_points_reprojected_paths = []
    val_ground_truth_paths = []
    val_ground_truth_interp_paths = []

    train_ids, val_ids = get_train_val_split_ids()

    n_scenes_to_process = min(args.n_scenes_to_process, MAX_SCENES)
    n_train = len([s for s in range(n_scenes_to_process) if s in train_ids])
    n_val = len([s for s in range(n_scenes_to_process) if s in val_ids])
    print('Total Scenes to process: {}'.format(n_scenes_to_process))
    print('Training: {}  Validation: {}'.format(n_train, n_val))

    # Add all tasks for processing each scene to pool inputs
    for scene_id in range(0, min(args.n_scenes_to_process, MAX_SCENES)):

        if scene_id in train_ids:
            tag = 'train'
        elif scene_id in val_ids:
            tag = 'val'
        else:
            raise ValueError('scene_id={} cannot be found in train or val split'.format(scene_id))

        current_scene = nusc.scene[scene_id]
        first_sample_token = current_scene['first_sample_token']
        last_sample_token = current_scene['last_sample_token']

        inputs = [
            tag,
            scene_id,
            get_closest_camera_image_to_lidar_scan_in_a_sample(current_scene),
            args.panoptic_seg_dir,
            first_sample_token,
            last_sample_token,
            args.n_forward_frames_to_reproject,
            args.n_backward_frames_to_reproject,
            args.nuscenes_data_derived_dirpath,
            args.paths_only
        ]

        pool_inputs.append(inputs)

        if not use_multithread:
            pool_results.append(process_scene(inputs))

    if use_multithread:
        # Create pool of threads
        with mp.Pool(args.n_thread) as pool:
            # Will fork n_thread to process scene
            pool_results = pool.map(process_scene, pool_inputs)

    # Unpack output paths
    for results in pool_results:

        tag, \
            camera_image_scene_paths, \
            lidar_scene_paths, \
            radar_points_scene_paths, \
            radar_points_reprojected_scene_paths, \
            ground_truth_scene_paths, \
            ground_truth_interp_scene_paths = results

        if tag == 'train':
            train_camera_image_paths.extend(camera_image_scene_paths)
            train_lidar_paths.extend(lidar_scene_paths)
            train_radar_points_paths.extend(radar_points_scene_paths)
            train_radar_points_reprojected_paths.extend(radar_points_reprojected_scene_paths)
            train_ground_truth_paths.extend(ground_truth_scene_paths)
            train_ground_truth_interp_paths.extend(ground_truth_interp_scene_paths)
        elif tag == 'val':
            val_camera_image_paths.extend(camera_image_scene_paths)
            val_lidar_paths.extend(lidar_scene_paths)
            val_radar_points_paths.extend(radar_points_scene_paths)
            val_radar_points_reprojected_paths.extend(radar_points_reprojected_scene_paths)
            val_ground_truth_paths.extend(ground_truth_scene_paths)
            val_ground_truth_interp_paths.extend(ground_truth_interp_scene_paths)
        else:
            raise ValueError('Found invalid tag: {}'.format(tag))

    # Subsample from validation set
    val_camera_image_subset_paths = val_camera_image_paths[::2]
    val_lidar_subset_paths = val_lidar_paths[::2]
    val_radar_points_subset_paths = val_radar_points_paths[::2]
    val_radar_points_reprojected_subset_paths = val_radar_points_reprojected_paths[::2]
    val_ground_truth_subset_paths = val_ground_truth_paths[::2]
    val_ground_truth_interp_subset_paths = val_ground_truth_interp_paths[::2]

    '''
    Write paths to file
    '''
    outputs = [
        [
            'training',
            [
                [
                    'image',
                    train_camera_image_paths,
                    TRAIN_IMAGE_FILEPATH
                ], [
                    'lidar',
                    train_lidar_paths,
                    TRAIN_LIDAR_FILEPATH
                ], [
                    'radar',
                    train_radar_points_paths,
                    TRAIN_RADAR_FILEPATH
                ], [
                    'radar reprojected',
                    train_radar_points_reprojected_paths,
                    TRAIN_RADAR_REPROJECTED_FILEPATH,
                ], [
                    'ground truth',
                    train_ground_truth_paths,
                    TRAIN_GROUND_TRUTH_FILEPATH
                ], [
                    'interpolated ground truth',
                    train_ground_truth_interp_paths,
                    TRAIN_GROUND_TRUTH_INTERP_FILEPATH
                ]
            ]
        ], [
            'validation',
            [
                [
                    'image',
                    val_camera_image_paths,
                    VAL_IMAGE_FILEPATH
                ], [
                    'lidar',
                    val_lidar_paths,
                    VAL_LIDAR_FILEPATH
                ], [
                    'radar',
                    val_radar_points_paths,
                    VAL_RADAR_FILEPATH
                ], [
                    'radar_reprojected',
                    val_radar_points_reprojected_paths,
                    VAL_RADAR_REPROJECTED_FILEPATH
                ], [
                    'ground truth',
                    val_ground_truth_paths,
                    VAL_GROUND_TRUTH_FILEPATH
                ], [
                    'interpolated ground truth',
                    val_ground_truth_interp_paths,
                    VAL_GROUND_TRUTH_INTERP_FILEPATH
                ], [
                    'image subset',
                    val_camera_image_subset_paths,
                    VAL_IMAGE_SUBSET_FILEPATH
                ], [
                    'lidar subset',
                    val_lidar_subset_paths,
                    VAL_LIDAR_SUBSET_FILEPATH
                ], [
                    'radar subset',
                    val_radar_points_subset_paths,
                    VAL_RADAR_SUBSET_FILEPATH
                ], [
                    'radar reprojected subset',
                    val_radar_points_reprojected_subset_paths,
                    VAL_RADAR_REPROJECTED_SUBSET_FILEPATH
                ], [
                    'ground truth subset',
                    val_ground_truth_subset_paths,
                    VAL_GROUND_TRUTH_SUBSET_FILEPATH
                ], [
                    'interpolated ground truth subset',
                    val_ground_truth_interp_subset_paths,
                    VAL_GROUND_TRUTH_INTERP_SUBSET_FILEPATH
                ]
            ]
        ]
    ]

    # Create output directories
    for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    for output_info in outputs:

        tag, output = output_info
        for output_type, paths, filepath in output:

            print('Storing {} {} {} file paths into: {}'.format(
                len(paths), tag, output_type, filepath))
            data_utils.write_paths(filepath, paths)
