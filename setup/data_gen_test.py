from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud, Box
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.neighbors import KDTree
import cv2
import h5py
from typing import NamedTuple
import pickle
from PIL import Image
import os.path as osp
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import points_in_box, view_points, box_in_image, BoxVisibility, transform_matrix
import multiprocessing as mp


# Create the nuScene object
nusc = NuScenes(version='v1.0-test', dataroot='../', verbose=True)
nusc_explorer = NuScenesExplorer(nusc)

max_scenes = 150

# Max number of threads to use for thread pool
n_thread = 20

# number of point closest to the radar input that you wnt in your ground truth. If the number of points is less than the number specified, we just copy the points again.
num_points = 100
n_forward = 2
n_backward = 2
pickle_file_name = "test_nuScenes_dataset_lidar_maps_interpolated_merged_2_2_100_with_filter.pkl"
# https://stackoverflow.com/questions/35988/c-like-structures-in-python

class Data_Struct(NamedTuple):
    scene_id: int
    sample_idx: int
    image_path: str
    ground_truth_points: np.ndarray
    input_points: np.ndarray
    lidar_depth_map_path: str
    
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
    return train_ids, val_ids
    
def my_map_pointcloud_to_image(pc,
                               pointsensor_token: str,
                               camera_token: str,
                               min_dist: float = 1.0,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels = None,
                               lidarseg_preds_bin_path: str = None,
                               show_panoptic: bool = False):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pc: Lidar point cloud.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
#     pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']

        gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
        semantic_table = getattr(nusc, gt_from)

        if lidarseg_preds_bin_path:
            sample_token = nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(nusc.dataroot,
                                                    nusc.get(gt_from, pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None

        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(lidarseg_labels_filename,
                                              filter_lidarseg_labels,
                                              nusc.lidarseg_name2idx_mapping,
                                              nusc.colormap)
            else:
                coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                    filter_lidarseg_labels,
                                                    nusc.lidarseg_name2idx_mapping,
                                                    nusc.colormap)

        else:
            coloring = depths
            print(f'Warning: There are no lidarseg labels in {nusc.version}. Points will be colored according '
                  f'to distance from the ego vehicle instead.')
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im

def my_map_image_to_pointcloud(pc,
                               pointsensor_token: str,
                               camera_token: str,
                               min_dist: float = 1.0):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pc: Lidar point cloud.
    :param pointsensor_token: Point sensor data token
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :return (pointcloud <np.float: 3, n)>).
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
#     pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))
    
    # First step: transform from camera into ego.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
    # Second step: transform from ego vehicle frame to global frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    
    # Third step: transform from global frame to ego frame
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # Fourth step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    return pc
    
def merge_lidar_point_clouds2(nusc, current_sample_token,n_forward,n_backward):
    """
    Merges Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.
    :param nusc: NuScenes Object.
    :param current_sample_token: (str) token for the current sample.
    :param n_forward: (int) number of frames to merge in the forward direction.
    :param n_backward: (int) number of frames to merge in the backward direction
    :return: <np.ndarray: image, np.ndarray: validity_map>.
    """
    nusc_explorer = NuScenesExplorer(nusc)
    my_sample = nusc.get('sample', current_sample_token) # get the sample
    lidar_token = my_sample['data']['LIDAR_TOP'] # get lidar token in the current sample
    main_camera_token = my_sample['data']['CAM_FRONT'] # get the camera token for the current sample
    # project the lidar frame into the camera frame
    main_points_lidar, main_coloring_lidar, main_image = nusc_explorer.map_pointcloud_to_image(pointsensor_token=lidar_token, camera_token=main_camera_token)
    main_image = np.asarray(main_image)
    lidar_image = np.zeros((main_image.shape[0], main_image.shape[1])) # create an empty lidar image
    validity_map = -1*np.ones((main_image.shape[0], main_image.shape[1])) # create a validity map to check which elements of the lidar image are valid
    for pt_idx in range(0,main_points_lidar.shape[1]):
        x = main_points_lidar[0,pt_idx]
        y = main_points_lidar[1,pt_idx]
        lidar_image[int(np.round(y)),int(np.round(x))] = main_coloring_lidar[pt_idx] # value of y,x is the depth
        validity_map[int(np.round(y)),int(np.round(x))] = 1
    n_f = 0 # count forward frames
    n_b = 0 # count backward frames
    while my_sample['next'] != "" and n_f < n_forward:
        next_sample_token = my_sample['next'] # got to the next sample
        next_sample = nusc.get('sample', next_sample_token) # get the sample
        next_lidar_token = next_sample['data']['LIDAR_TOP'] # get lidar token in the current sample
        next_camera_token = next_sample['data']['CAM_FRONT'] # get lidar token in the current sample
        next_lidar_data = nusc.get('sample_data', next_lidar_token) # get lidar data in the current sample
        next_pcl_path = os.path.join(nusc.dataroot, next_lidar_data['filename']) # get file name where lidar point clouds are saved
        next_lpc = LidarPointCloud.from_file(next_pcl_path) # get the lidar point cloud
        lidar_points = next_lpc.points
        data_path, boxes, cam_intrinsic = nusc.get_sample_data(next_camera_token, box_vis_level=BoxVisibility.ANY,use_flat_vehicle_coordinates=False) # get all bounding boxes for the lidar data in the current sample
        next_points_lidar, next_coloring_lidar, _ = nusc_explorer.map_pointcloud_to_image(pointsensor_token=next_lidar_token, camera_token=next_camera_token)
        lidar_projected_image = np.zeros((900,1600)) # project the lidar points to image plane
        for idx in range(0,next_points_lidar.shape[-1]):
            x,y = next_points_lidar[0:2,idx]
            lidar_projected_image[int(np.round(y)),int(np.round(x))] = next_coloring_lidar[idx]
        for box in boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=cam_intrinsic, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:,0]))
                min_y = int(np.min(corners.T[:,1]))
                max_x = int(np.max(corners.T[:,0]))
                max_y = int(np.max(corners.T[:,1]))
                lidar_projected_image[min_y:max_y,min_x:max_x] = 0 # filter out the points inside the bounding box
        # now we need to go back from the image frame to the lidar point cloud frame
        lidar_points_y, lidar_points_x  = np.nonzero(lidar_projected_image)
        lidar_points_z = lidar_projected_image[lidar_points_y, lidar_points_x]
        x_y_homogeneous = np.stack((lidar_points_x,lidar_points_y,np.ones_like(lidar_points_x)),axis=0)
        x_y_lifted = np.matmul(np.linalg.inv(cam_intrinsic),x_y_homogeneous)
        x_y_z = x_y_lifted*np.expand_dims(lidar_points_z,axis=0)
        # to convert the lidar point cloud into a LidarPointCloud object, we need 4,n shape. So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array,axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis = 0)

        # convert lidar point cloud into a nuScene LidarPointCloud object
        xyz_lpc = LidarPointCloud(x_y_z)
        
        # now we can transform the points back to the lidar frame of reference
        transformed_point_cloud = my_map_image_to_pointcloud(xyz_lpc, pointsensor_token=next_lidar_token, camera_token=next_camera_token)
                 
        next_points_lidar, next_coloring_lidar, _ = my_map_pointcloud_to_image(pc=transformed_point_cloud,pointsensor_token=next_lidar_token, camera_token=main_camera_token)
        for pt_idx in range(0,next_points_lidar.shape[1]):
            x = next_points_lidar[0,pt_idx]
            y = next_points_lidar[1,pt_idx]
            if next_coloring_lidar[pt_idx] < lidar_image[int(np.round(y)),int(np.round(x))] and validity_map[int(np.round(y)),int(np.round(x))] == 1:
                lidar_image[int(np.round(y)),int(np.round(x))] = next_coloring_lidar[pt_idx] # we only take the depth if it is not occluded
            elif validity_map[int(np.round(y)),int(np.round(x))] != 1:
                lidar_image[int(np.round(y)),int(np.round(x))] = next_coloring_lidar[pt_idx]
                validity_map[int(np.round(y)),int(np.round(x))] = 1
        n_f = n_f + 1
    while my_sample['prev'] != "" and n_b < n_backward:
        prev_sample_token = my_sample['prev'] # got to the prev sample
        prev_sample = nusc.get('sample', prev_sample_token) # get the sample
        prev_lidar_token = prev_sample['data']['LIDAR_TOP'] # get lidar token in the current sample
        prev_camera_token = prev_sample['data']['CAM_FRONT'] # get lidar token in the current sample
        prev_lidar_data = nusc.get('sample_data', prev_lidar_token) # get lidar data in the current sample
        prev_pcl_path = os.path.join(nusc.dataroot, prev_lidar_data['filename']) # get file name where lidar point clouds are saved
        prev_lpc = LidarPointCloud.from_file(prev_pcl_path) # get the lidar point cloud
        lidar_points = prev_lpc.points
        data_path, boxes, cam_intrinsic = nusc.get_sample_data(prev_camera_token, box_vis_level=BoxVisibility.ANY,use_flat_vehicle_coordinates=False) # get all bounding boxes for the lidar data in the current sample
        prev_points_lidar, prev_coloring_lidar, _ = nusc_explorer.map_pointcloud_to_image(pointsensor_token=prev_lidar_token, camera_token=prev_camera_token)
        lidar_projected_image = np.zeros((900,1600)) # project the lidar points to image plane
        for idx in range(0,prev_points_lidar.shape[-1]):
            x,y = prev_points_lidar[0:2,idx]
            lidar_projected_image[int(np.round(y)),int(np.round(x))] = prev_coloring_lidar[idx]
        for box in boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=cam_intrinsic, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:,0]))
                min_y = int(np.min(corners.T[:,1]))
                max_x = int(np.max(corners.T[:,0]))
                max_y = int(np.max(corners.T[:,1]))
                lidar_projected_image[min_y:max_y,min_x:max_x] = 0 # filter out the points inside the bounding box
        # now we need to go back from the image frame to the lidar point cloud frame
        lidar_points_y, lidar_points_x  = np.nonzero(lidar_projected_image)
        lidar_points_z = lidar_projected_image[lidar_points_y, lidar_points_x]
        x_y_homogeneous = np.stack((lidar_points_x,lidar_points_y,np.ones_like(lidar_points_x)),axis=0)
        x_y_lifted = np.matmul(np.linalg.inv(cam_intrinsic),x_y_homogeneous)
        x_y_z = x_y_lifted*np.expand_dims(lidar_points_z,axis=0)
        # to convert the lidar point cloud into a LidarPointCloud object, we need 4,n shape. So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array,axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis = 0)

        # convert lidar point cloud into a nuScene LidarPointCloud object
        xyz_lpc = LidarPointCloud(x_y_z)
        
        # now we can transform the points back to the lidar frame of reference
        transformed_point_cloud = my_map_image_to_pointcloud(xyz_lpc, pointsensor_token=prev_lidar_token, camera_token=prev_camera_token)
                 
        prev_points_lidar, prev_coloring_lidar, _ = my_map_pointcloud_to_image(pc=transformed_point_cloud,pointsensor_token=prev_lidar_token, camera_token=main_camera_token)
        for pt_idx in range(0,prev_points_lidar.shape[1]):
            x = prev_points_lidar[0,pt_idx]
            y = prev_points_lidar[1,pt_idx]
            if prev_coloring_lidar[pt_idx] < lidar_image[int(np.round(y)),int(np.round(x))] and validity_map[int(np.round(y)),int(np.round(x))] == 1:
                lidar_image[int(np.round(y)),int(np.round(x))] = prev_coloring_lidar[pt_idx] # we only take the depth if it is not occluded
            elif validity_map[int(np.round(y)),int(np.round(x))] != 1:
                lidar_image[int(np.round(y)),int(np.round(x))] = prev_coloring_lidar[pt_idx]
                validity_map[int(np.round(y)),int(np.round(x))] = 1
        n_b = n_b + 1
    # need to convert this to the same format used by nuScenes to return Lidar points
    temp_return = np.asarray(np.where(lidar_image))
    return_points_lidar = np.empty(temp_return.shape)
    return_points_lidar[0,:] = temp_return[1,:]
    return_points_lidar[1,:] = temp_return[0,:]
    return_coloring_lidar = lidar_image[lidar_image!=0]
    return return_points_lidar, return_coloring_lidar
    
### Total scenes = 850
def point_registration(points_lidar, points_radar, coloring_radar, coloring_lidar):
    num_points = len(points_radar[0])
    num_points2 = len(points_lidar[0])
    X = np.empty([num_points2,2])
    X[:,0] = points_lidar[0][:]
    X[:,1] = coloring_lidar*100
    Xp = np.empty([num_points,2])
    Xp[:,0] = points_radar[0][:]
    Xp[:,1] = coloring_radar*100
    
    tree = KDTree(X)
    
    lidar_corr_x = []
    lidar_corr_y = []
    radar_corr_x = []
    radar_corr_y = []
    radar_corr_z = []
    idx_array = []
    count = 0
    for i in range(0,Xp.shape[0]):
        radar_point = np.expand_dims(Xp[i],0)
        index, distance = tree.query_radius(radar_point, r=25, count_only=False, return_distance=True, sort_results=True)
        if len(index[0]) == 0:
            count = count + 1
            continue
        lidar_correspondence = X[index[0][0]]
#         print(radar_point,points_radar[1][i])
#         print(lidar_correspondence,points_lidar[1][index[0][0]])
#         print(distance[0][0])
        for idx11 in range(0,len(index[0])):
            lidar_corr_x.append(X[index[0][idx11]][0])
            lidar_corr_y.append(points_lidar[1][index[0][idx11]])
            radar_corr_x.append(radar_point[0][0])
            radar_corr_y.append(points_radar[1][i])
            radar_corr_z.append(coloring_radar[i])
            idx_array.append(i)
    radar_points_img = np.stack([radar_corr_x, radar_corr_y, radar_corr_z],axis=-1) # original radar points with garbage y
    radar_pts_no_z = np.stack([radar_corr_x, radar_corr_y],axis=-1)
    lidar_points_img = np.stack([lidar_corr_x, lidar_corr_y],axis=-1) # points that are ground truth
    shift = []
    for k in range(0,len(lidar_points_img)):
        shift.append(lidar_points_img[k] - radar_pts_no_z[k]) 
#     print(count)
#     print(num_points)
    return lidar_points_img, radar_points_img, idx_array, shift

def check_for_length_and_copy(input_array, final_len):
    if len(input_array) > final_len:
        input_array = input_array[:final_len]
    elif len(input_array) < final_len:
        while len(input_array) < final_len:
            input_array.append(input_array[0])
    input_array = np.asarray(input_array)
    return np.asarray(input_array)

def reorganize_points(radar_points, lidar_points, shift_points, idx_array, num_points):
    radar_list = []
    lidar_list = []
    shift_list = []
    radar_list_big = []
    lidar_list_big = []
    shift_list_big = []
    flag = idx_array[0]
    for i in range(0,len(idx_array)):
        if idx_array[i] == flag:
            radar_to_go = [radar_points[i][0], radar_points[i][1], radar_points[i][2]]
            lidar_to_go = [lidar_points[i][0], lidar_points[i][1]]
            shift_to_go = [shift_points[i][0], shift_points[i][1]]
            radar_list.append(radar_to_go)
            lidar_list.append(lidar_to_go)
            shift_list.append(shift_to_go)
        elif idx_array[i] != flag:
            flag = idx_array[i]
            if len(radar_list_big) == 0:
                radar_list = check_for_length_and_copy(radar_list, num_points)
                lidar_list = check_for_length_and_copy(lidar_list, num_points)
                shift_list = check_for_length_and_copy(shift_list, num_points)
                radar_list_big = radar_list
                lidar_list_big = lidar_list
                shift_list_big = shift_list
            else:
                radar_list = check_for_length_and_copy(radar_list, num_points)
                lidar_list = check_for_length_and_copy(lidar_list, num_points)
                shift_list = check_for_length_and_copy(shift_list, num_points)
                radar_list = radar_list.reshape((1,num_points,3))
                lidar_list = lidar_list.reshape((1,num_points,2))
                shift_list = shift_list.reshape((1,num_points,2))
                if radar_list_big.shape == (num_points,3):
                    radar_list_big = radar_list.reshape((1,num_points,3))
                    lidar_list_big = lidar_list.reshape((1,num_points,2))
                    shift_list_big = shift_list.reshape((1,num_points,2))
                radar_list_big = np.concatenate((radar_list_big,radar_list), axis=0)
                lidar_list_big = np.concatenate((lidar_list_big,lidar_list), axis=0)
                shift_list_big = np.concatenate((shift_list_big,shift_list), axis=0)
            radar_list = []
            lidar_list = []
            shift_list = []
            radar_to_go = [radar_points[i][0], radar_points[i][1], radar_points[i][2]]
            lidar_to_go = [lidar_points[i][0], lidar_points[i][1]]
            shift_to_go = [shift_points[i][0], shift_points[i][1]]
            radar_list.append(radar_to_go)
            lidar_list.append(lidar_to_go)
            shift_list.append(shift_to_go)
    if len(radar_list_big) == 0:
        radar_list = check_for_length_and_copy(radar_list, num_points)
        lidar_list = check_for_length_and_copy(lidar_list, num_points)
        shift_list = check_for_length_and_copy(shift_list, num_points)
        radar_list_big = radar_list
        lidar_list_big = lidar_list
        shift_list_big = shift_list
    else:
        radar_list = check_for_length_and_copy(radar_list, num_points)
        lidar_list = check_for_length_and_copy(lidar_list, num_points)
        shift_list = check_for_length_and_copy(shift_list, num_points)
        radar_list = radar_list.reshape((1,num_points,3))
        lidar_list = lidar_list.reshape((1,num_points,2))
        shift_list = shift_list.reshape((1,num_points,2))
        if radar_list_big.shape == (num_points,3):
            radar_list_big = radar_list.reshape((1,num_points,3))
            lidar_list_big = lidar_list.reshape((1,num_points,2))
            shift_list_big = shift_list.reshape((1,num_points,2))
        radar_list_big = np.concatenate((radar_list_big,radar_list), axis=0)
        lidar_list_big = np.concatenate((lidar_list_big,lidar_list), axis=0)
        shift_list_big = np.concatenate((shift_list_big,shift_list), axis=0)
    return radar_list_big, lidar_list_big, shift_list_big

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
        radar_token = my_sample['data']['RADAR_FRONT']
        lidar_token = my_sample['data']['LIDAR_TOP']
        camera_token = my_sample['data']['CAM_FRONT']
        radar_sample = nusc.get('sample_data', radar_token)
        camera_sample = nusc.get('sample_data', camera_token)
        lidar_sample = nusc.get('sample_data', lidar_token)
        
        ## Get Radar Data
        pcl_path = os.path.join(nusc.dataroot, radar_sample['filename'])
        RadarPointCloud.disable_filters()
        rpc = RadarPointCloud.from_file(pcl_path)
        
        assert (rpc.points[2, :] == 0).all()
        
        # Get Lidar Data
        pcl_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
        lpc = LidarPointCloud.from_file(pcl_path)
        
        # Get Camera Data
        camera_image = image.imread(os.path.join(nusc.dataroot, camera_sample['filename']))
        
        # Transform Lidar and Radar Points to the image coordinate
        points_radar, coloring_radar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=radar_sample['token'],
            camera_token=camera_sample['token'])
        
        # Merges n_forward and n_backward number of point clouds to frame at sample token
        points_lidar, coloring_lidar = merge_lidar_point_clouds2(
            nusc, 
            sample_token,
            n_forward=n_forward,
            n_backward=n_backward)
        
        # Perform registration: maps radar point to ground truth lidar points
        ground_truth_points, input_points, idx_array, shift = point_registration(
            points_lidar, 
            points_radar,
            coloring_radar, 
            coloring_lidar)
        
        if len(idx_array) == 0:
            print('Found empty scene, skipping scene: scene_id={}'.format(scene_id))
            break
            
        # Find num_points number of points closest to input points using KDTree
        input_points, ground_truth_points, shift = reorganize_points(
            input_points,
            ground_truth_points, 
            shift, 
            idx_array, 
            num_points)
        
        # Some sample may only 1 radar point, which drops a dimension to (N, 2) 
        # where N (num_points) number of ground truth points
        if ground_truth_points.ndim == 2:
            # We add it back so it is (1, N, 2) for one radar point
            ground_truth_points = np.expand_dims(ground_truth_points, 0)
            
        # Use 2 to denote class of all invalid points
        lidar_frame = np.full(camera_image.shape[0:2], 2)

        # Iterate through N lidar points 
        for idx in range(0, points_lidar.shape[-1]):
            # Fetch x, y coordinate and quantize to pixel coordinate
            x, y = points_lidar[0:2, idx]
            lidar_frame[int(np.round(y)),int(np.round(x))] = 0
        
        # Make lidar frame H x W x 1 and repeat the frame N times
        lidar_frames = np.expand_dims(lidar_frame, -1)
        lidar_frames = np.tile(lidar_frames, (1, 1, ground_truth_points.shape[0]))
        
        # For every ground truth lidar point and radar point
        for idx_channel in range(0, ground_truth_points.shape[0]):
            for idx_point in range(0, ground_truth_points.shape[1]):
                # We mark all ground truth matching points based on closness to 1
                x, y = ground_truth_points[idx_channel, idx_point, :]
                lidar_frames[int(np.round(y)), int(np.round(x)),idx_channel] = 1
        
        # Store ground truth label (0, 1, 2), 0 too far, 1 close, 2 invalid
        lidar_frames = np.uint8(lidar_frames)
        
        dir_path, file_name = os.path.split(camera_sample['filename'])
        dir_path = dir_path.replace('samples','pseudo_ground_truth_test')
        file_name = os.path.splitext(file_name)[0] + '-{}.png'
        
        output_path = os.path.join(dir_path, file_name)
        
        # In case multiple threads create same directory
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except:
                pass
        
        # Save each ground truth label map as a PNG file
        for idx_channel in range(0, ground_truth_points.shape[0]):
            output_path_point = output_path.format(idx_channel)
            output_png = Image.fromarray(np.squeeze(lidar_frames[:, :, idx_channel]), mode='L')
            output_png.save(output_path_point)
        
        data_item = Data_Struct(scene_id, sample_id, camera_sample['filename'], shift, input_points, output_path)
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

pool_inputs = []

# Add all tasks for processing each scene to pool inputs
for scene_id in range(0, max_scenes):
    my_scene = nusc.scene[scene_id]
    first_sample_token = my_scene['first_sample_token']
    last_sample_token = my_scene['last_sample_token']
    
    pool_inputs.append((scene_id, first_sample_token, last_sample_token))
    
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