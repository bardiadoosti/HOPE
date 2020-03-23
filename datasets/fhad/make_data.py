import os
import numpy as np
import trimesh


# Loading utilities
def load_objects(obj_root):
    object_names = ['juice_bottle', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = mesh
    return all_models


def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    #print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1)[sample['frame_idx']]
    return skeleton


def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                            sample['seq_idx'], 'object_pose.txt')
    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample['frame_idx']]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    #print('Loading obj transform from {}'.format(seq_path))
    return trans_matrix


# Change this path
root = '/path/to/FHAD/hand_pose_action'
skeleton_root = os.path.join(root, 'Hand_pose_annotation_v1')
obj_root = os.path.join(root, 'Object_models')
obj_trans_root = os.path.join(root, 'Object_6D_pose_annotation_v1_1')
file_root = os.path.join(root, 'Video_files')
# Load object mesh
object_infos = load_objects(obj_root)
reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                     [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                     [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                     [0, 0, 0, 1]])
cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030],
                     [0, 0, 1]])


images_train = []
points2d_train = []
points3d_train = []

images_val = []
points2d_val = []
points3d_val = []

images_test = []
points2d_test = []
points3d_test = []

for subject in os.listdir(obj_trans_root):
    print(subject)
    for action_name in os.listdir(os.path.join(obj_trans_root, subject)):
        print(action_name)
        obj = '_'.join(action_name.split('_')[1:])
        for seq_idx in os.listdir(os.path.join(obj_trans_root, subject, action_name)):
            sset = 'train'
            if seq_idx == '1':
                sset = 'val'
            elif seq_idx == '3':
                sset = 'test'
            try:
                for file_name in os.listdir(os.path.join(file_root, subject, action_name, seq_idx, 'color')):
                    frame_idx = int(file_name.split('.')[0].split('_')[1])
                    sample = {
                        'subject': subject,
                        'action_name': action_name,
                        'seq_idx': seq_idx,
                        'frame_idx': frame_idx,
                        'object': obj
                    }
                    img_path = os.path.join(file_root, subject, action_name, seq_idx, 'color', file_name)
        
                    skel = get_skeleton(sample, skeleton_root)[reorder_idx]
    
                    # Load object transform
                    obj_trans = get_obj_transform(sample, obj_trans_root)
                
                    mesh = object_infos[sample['object']]
                    verts = np.array(mesh.bounding_box_oriented.vertices) * 1000
                
                    # Apply transform to object
                    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
                    verts_trans = obj_trans.dot(hom_verts.T).T
                
                    # Apply camera extrinsic to object
                    verts_camcoords = cam_extr.dot(verts_trans.transpose()).transpose()[:, :3]
                    # Project and object skeleton using camera intrinsics
                    verts_hom2d = np.array(cam_intr).dot(verts_camcoords.transpose()).transpose()
                    verts_proj = (verts_hom2d / verts_hom2d[:, 2:])[:, :2]
                    
                    # Apply camera extrinsic to hand skeleton
                    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
                    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
                    
                    skel_hom2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
                    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]
                
                    points = np.concatenate((skel_camcoords, verts_camcoords))
                    projected_points = np.concatenate((skel_proj, verts_proj))
                    
                    if sset == 'train':
                        images_train.append(img_path)
                        points2d_train.append(projected_points)
                        points3d_train.append(points)
                    if sset == 'val':
                        images_val.append(img_path)
                        points2d_val.append(projected_points)
                        points3d_val.append(points)
                    if sset == 'test':
                        images_test.append(img_path)
                        points2d_test.append(projected_points)
                        points3d_test.append(points)
            except:
                print('====%s, %s, %s===='%(subject, action_name, seq_idx))

np.save('./images-train.npy', np.array(images_train))
np.save('./points2d-train.npy', np.array(points2d_train))
np.save('./points3d-train.npy', np.array(points3d_train))

np.save('./images-val.npy', np.array(images_val))
np.save('./points2d-val.npy', np.array(points2d_val))
np.save('./points3d-val.npy', np.array(points3d_val))

np.save('./images-test.npy', np.array(images_test))
np.save('./points2d-test.npy', np.array(points2d_test))
np.save('./points3d-test.npy', np.array(points3d_test))