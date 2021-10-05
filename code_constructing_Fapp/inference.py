import numpy as np
from scipy.interpolate import griddata
import argparse
import os

def to_image(prokected_points, h, w):
    '''
    transform the center to (0,0)
    '''
    image_vertices = prokected_points.copy()
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    image_vertices[:,1] = h - image_vertices[:,1] - 1
    return image_vertices

def orthogonal_transform(points3D, scale, R, t):
    '''
    orthogonal transform
    '''
    t3d = np.squeeze(np.array(t, dtype = np.float32))
    transformed_vertices = scale * points3D.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices

def project_to_image(points3D, scale, R, t, h, w):
    '''
    project 3D points to 2d plane in orthogonal projection
    '''
    prokected_points = orthogonal_transform(points3D, scale, R, t)
    prokected_points = to_image(prokected_points, h, w)
    return prokected_points

def compute_projected_mesh_points(shape_para,exp_para,R,T,scale,image_size,model_3dmm):
    '''
    compute the projected mesh points from facial animation parameters
    :param shape_para: the shape parameter of reference face
    :param scale: the scale parameter in orthogonal projection
    :param exp_para: the expression parameter in orthogonal projection
    :param R: the head rotation in orthogonal projection
    :param T: the head translation in orthogonal projection
    :param image_size: the size of image
    :param model_3dmm: 3dmm model
    :return: projected_mesh_points
    '''
    shape_para = np.expand_dims(shape_para, 1)
    exp_para = np.expand_dims(exp_para, 1)
    R_matrix = R.reshape((3, 3))
    ## compute 3d mesh points in 3DMM
    mesh_points_3D = model_3dmm.generate_vertices(shape_para, exp_para)
    ## project 3D points to 2D plane
    projected_2Dpoints = project_to_image(mesh_points_3D, scale, R_matrix, T, image_size[0], image_size[1])
    return projected_2Dpoints


def make_coordinate_grid(image_size):
    h, w = image_size
    x = np.arange(w)
    y = np.arange(h)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    xx = x.reshape(1, -1).repeat(h, axis=0)
    yy = y.reshape(-1, 1).repeat(w, axis=1)
    meshed = np.stack([xx, yy], 2)
    return meshed

def construct_Fapp(reference_projected_mesh_points,
                   drive_projected_mesh_points,image_size):
    '''
    compute Fapp from projected mesh points
    reference_projected_mesh_points: the projected mesh points of reference image
    drive_projected_mesh_points: the driving projected mesh points
    '''
    ## resize to -1 ~ 1
    reference_projected_mesh_points = (reference_projected_mesh_points / image_size * 2) - 1
    ### compute the max heigh of face
    face_max_h = np.max(drive_projected_mesh_points[:, 1]).astype(np.int)
    ## resize to -1 ~ 1
    drive_projected_mesh_points = (drive_projected_mesh_points / image_size * 2) - 1
    drive_projected_mesh_points_yx = drive_projected_mesh_points[:, [1, 0]]
    ## compute sparse dense flow
    sparse_dense_flow = reference_projected_mesh_points - drive_projected_mesh_points
    ## compute average head motion
    mean_dense_flow = np.mean(sparse_dense_flow, axis=0)
    ## compute the dense flow in head-related region
    grid_nums = complex(str(image_size) + "j")
    grid_y, grid_x = np.mgrid[-1:1:grid_nums, -1:1:grid_nums]
    dense_foreground_flow_x = griddata(drive_projected_mesh_points_yx, sparse_dense_flow[:, 0], (grid_y, grid_x), method='nearest')
    ## compute dense flow in torso related region
    dense_foreground_flow_x[face_max_h:, :] = mean_dense_flow[0]
    dense_foreground_flow_y = griddata(drive_projected_mesh_points_yx, sparse_dense_flow[:, 1], (grid_y, grid_x), method='nearest')
    dense_foreground_flow_y[face_max_h:, :] = mean_dense_flow[1]
    Fapp = np.stack([dense_foreground_flow_x, dense_foreground_flow_y], 2)
    ## transform into grid data
    grid_mesh = make_coordinate_grid((image_size,image_size))
    Fapp = grid_mesh + Fapp

    return Fapp

def parse_opts():
    parser = argparse.ArgumentParser(description='construct Fapp')
    parser.add_argument('--reference_projected_mesh_points_path', type=str,
                        default='./test_data/taile_source_points.npy',
                        help='the projected mesh points of reference image')
    parser.add_argument('--drive_projected_mesh_points_path', type=str,
                        default='./test_data/taile_drive_points.npy',
                        help='the driving projected mesh points')
    parser.add_argument('--image_size', type=int, default=512, help='the size of image')
    parser.add_argument('--res_dir', type=str,default='./result',help='the dir of results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''
    It is not allowed to share the 3DMM model, so we release the inference code
    of constructing Fapp from projected mesh points.the function 
    "compute_projected_mesh_points" shows how to compute 
    projected mesh points from facial animation parameters. 
    '''
    opt = parse_opts()
    reference_projected_mesh_points = np.load(opt.reference_projected_mesh_points_path)
    drive_projected_mesh_points = np.load(opt.drive_projected_mesh_points_path)
    frame_num = drive_projected_mesh_points.shape[0]
    res_Fapp = []
    for i in range(frame_num):
        print('construct {}/{} Fapp'.format(i,frame_num))
        Fapp_i = construct_Fapp(reference_projected_mesh_points,
                   drive_projected_mesh_points[i,:,:],opt.image_size)
        res_Fapp.append(Fapp_i)
    res_Fapp = np.stack(res_Fapp,0)
    res_Fapp_path = os.path.join(opt.res_dir,os.path.basename(opt.reference_projected_mesh_points_path).replace('_source_points','_Fapp'))
    np.save(res_Fapp_path,res_Fapp)
    