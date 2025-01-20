from mujoco_py.generated import const
from metaworld import policies
import numpy as np
import cv2
import os


def get_policy(env_name):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except Exception as e:
        print(e)
        policy = None
    return policy

def get_robot_seg(env):
    seg = env.render(segmentation=True)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])

    for i in geoms_ids:
        if 2 <= i <= 33:
            img[ids == i] = True
    return img

def get_geom_ids(env):
    seg = env.render(segmentation=True)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])
    return geoms_ids

def get_seg(env, camera, resolution, seg_ids):
    seg = env.render(segmentation=True, resolution=resolution, camera_name=camera)
    img = np.zeros(seg.shape[:2], dtype=bool)
    types = seg[:, :, 0]
    ids = seg[:, :, 1]
    geoms = types == const.OBJ_GEOM
    geoms_ids = np.unique(ids[geoms])
    # print(geoms_ids)

    for i in geoms_ids:
        if i in seg_ids:
            img[ids == i] = True
    img = img.astype('uint8') * 255
    return cv2.medianBlur(img, 3)

def enumerate_segs(env, camera, resolution, out_dir="./part_segs"):
    os.makedirs(out_dir, exist_ok=True)
    geom_ids = get_geom_ids(env)
    frame, _ = env.render(depth=True, offscreen=True, camera_name=camera, resolution=resolution)
    for i in geom_ids:
        seg = get_seg(env, camera, resolution, [i])
        masked_image = cv2.bitwise_and(frame, frame, mask=seg)
        # convert to BGR
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out_dir}/{i}.png", masked_image)


def get_cmat(env, cam_name, resolution):
    id = env.sim.model.camera_name2id(cam_name)
    fov = env.sim.model.cam_fovy[id]
    pos = env.sim.data.cam_xpos[id]
    rot = env.sim.data.cam_xmat[id].reshape(3, 3).T
    width, height = resolution
    # Translation matrix (4x4).
    translation = np.eye(4)
    translation[0:3, 3] = -pos
    # Rotation matrix (4x4).
    rotation = np.eye(4)
    rotation[0:3, 0:3] = rot
    # Focal transformation matrix (3x4).
    focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0 # focal length
    focal = np.diag([-focal_scaling, -focal_scaling, 1.0, 0])[0:3, :]
    # Image matrix (3x3).
    image = np.eye(3)
    image[0, 2] = (width - 1) / 2.0
    image[1, 2] = (height - 1) / 2.0
    
    return image @ focal @ rotation @ translation # 3x4

def get_obj_uvd(env, cam_name, resolution, obj_ids, far_plane=10):
    _, depth = env.render(depth=True, offscreen=True, camera_name=cam_name, resolution=resolution)
    seg = get_seg(env, cam_name, resolution, obj_ids)
    X, Y = np.meshgrid(np.arange(resolution[0]), np.arange(resolution[1]))
    uvd_map = np.stack([X*depth, Y*depth, depth], axis=2)
    uvd_points = uvd_map[(seg>0) * (depth*-1<far_plane)]
    return uvd_points

def get_obj_xyz(env, cam_name, resolution, obj_ids, far_plane=10):
    cmat = get_cmat(env, cam_name, resolution)
    cmat_ext = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    cmat_inv = np.linalg.inv(cmat_ext)
    uvd_points = get_obj_uvd(env, cam_name, resolution, obj_ids, far_plane)
    uvd_normalized = uvd_points[:, :2] / uvd_points[:, 2:]
    uvd_normalized = np.concatenate([uvd_normalized, uvd_points[:, 2:]], axis=1)
    uvd_points_ext = np.concatenate([uvd_points, np.ones((uvd_points.shape[0], 1))], axis=1)
    xyz_points = (cmat_inv @ uvd_points_ext.T)[:3].T
    return xyz_points, uvd_normalized

def get_obj_uvd_all(env, cam_name, resolution, far_plane=10):
    geom_ids = get_geom_ids(env)
    points = []
    seg_ids = []
    _, depth = env.render(depth=True, offscreen=True, camera_name=cam_name, resolution=resolution)
    for id in geom_ids:
        seg = get_seg(env, cam_name, resolution, [id])
        X, Y = np.meshgrid(np.arange(resolution[0]), np.arange(resolution[1]))
        uvd_map = np.stack([X*depth, Y*depth, depth], axis=2)
        uvd_points = uvd_map[(seg>0) * (depth*-1<far_plane)]
        points.append(uvd_points)
        seg_ids += [id] * len(uvd_points)
    return points, seg_ids

def get_obj_xyz_all(env, cam_name, resolution, far_plane=10):
    cmat = get_cmat(env, cam_name, resolution)
    cmat_ext = np.concatenate([cmat, np.array([[0, 0, 0, 1]])], axis=0)
    cmat_inv = np.linalg.inv(cmat_ext)
    uvd_points, seg_ids = get_obj_uvd_all(env, cam_name, resolution, far_plane)
    uvd_points = np.concatenate(uvd_points, axis=0)
    uvd_normalized = uvd_points[:, :2] / uvd_points[:, 2:]
    uvd_normalized = np.concatenate([uvd_normalized, uvd_points[:, 2:]], axis=1)
    uvd_points_ext = np.concatenate([uvd_points, np.ones((uvd_points.shape[0], 1))], axis=1)
    xyz_points = (cmat_inv @ uvd_points_ext.T)[:3].T
    return xyz_points, uvd_normalized, seg_ids

def xyz_to_uv(env, cam_name, xyz, resolution=(640, 480)):
    cmat = get_cmat(env, cam_name, resolution)
    xyz_ext = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    udvdd = (cmat @ xyz_ext.T).T
    uv = udvdd[:, :2] / udvdd[:, 2:]
    uvd = np.concatenate([uv, udvdd[:, 2:]], axis=1)
    return uv, uvd

def xyz_global_to_local(env, xyz, cam_name, object_id):
    pos = env.sim.data.geom_xpos[object_id]
    rot = env.sim.data.geom_xmat[object_id].reshape(3, 3)
    inv_rot = np.linalg.inv(rot)
    return (inv_rot @ (xyz.T - pos[:, None])).T

def xyz_local_to_global(env, xyz, cam_name, object_id):
    pos = env.sim.data.geom_xpos[object_id]
    rot = env.sim.data.geom_xmat[object_id].reshape(3, 3)
    return (rot @ xyz.T + pos[:, None]).T

def xyz_global_to_local_all(env, xyz, cam_name, seg_ids):
    seg_ids_set = list(set(seg_ids))
    xyz_local = []
    for seg_id in seg_ids_set:
        xyz_local.append(xyz_global_to_local(env, xyz[seg_ids == seg_id], cam_name, seg_id))
    return np.concatenate(xyz_local, axis=0)

def xyz_local_to_global_all(env, xyz, cam_name, seg_ids):
    seg_ids_set = list(set(seg_ids))
    xyz_global = []
    for seg_id in seg_ids_set:
        xyz_global.append(xyz_local_to_global(env, xyz[seg_ids == seg_id], cam_name, seg_id))
    return np.concatenate(xyz_global, axis=0)

def construct_extrinsic_matrix(pos, rot):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic[:3, 3] = pos
    return extrinsic


class FlowManager():
    def __init__(self, env, object_ids=[0], resolution=(320, 240), camera_name='corner', record_depths=False):
        self.flows = []
        self.env = env
        self.object_ids = object_ids
        self.resolution = resolution
        self.camera_name = camera_name
        self.record_depths = record_depths
        if record_depths:
            self.depths = {
                "before": [],
                "after": []
            }
        self.rel_transforms = []
        self.update_object_attr()

    def update_object_attr(self):
        xyz_global, uvd = get_obj_xyz(self.env, self.camera_name, self.resolution, self.object_ids)
        self.obj_xyz_local = xyz_global_to_local(self.env, xyz_global, self.camera_name, self.object_ids[0])
        if self.record_depths:
            self.last_depth = uvd
        self.obj_uvs, _ = xyz_to_uv(self.env, self.camera_name, xyz_global, self.resolution)
        self.last_pos = self.env.sim.data.geom_xpos[self.object_ids[0]].copy()
        self.last_rot = self.env.sim.data.geom_xmat[self.object_ids[0]].reshape(3, 3).copy()
        self.last_xyz_global = xyz_global.copy()
    
    def append_flow(self):
        current_xyz_global = xyz_local_to_global(self.env, self.obj_xyz_local, self.camera_name, self.object_ids[0])
        current_uv, current_d = xyz_to_uv(self.env, self.camera_name, current_xyz_global, self.resolution)
        if self.record_depths:
            self.depths["before"].append(self.last_depth)
            self.depths["after"].append(current_d)
            
        ### calculate extrinsic matrix
        last_ext = construct_extrinsic_matrix(self.last_pos, self.last_rot)
        current_ext = construct_extrinsic_matrix(self.env.sim.data.geom_xpos[self.object_ids[0]], self.env.sim.data.geom_xmat[self.object_ids[0]].reshape(3, 3))
        rel_ext = current_ext @ np.linalg.inv(last_ext)
        self.rel_transforms.append(rel_ext)

        # ### make sure that the gt transform works
        # print("-----------")
        # print(self.last_xyz_global[:5])
        # print(current_xyz_global[:5])
        # print((rel_ext @ np.concatenate([self.last_xyz_global.T, np.ones((1, len(self.last_xyz_global)))]))[:3].T[:5])
        # assert np.allclose(current_xyz_global, (rel_ext @ np.concatenate([self.last_xyz_global.T, np.ones((1, len(self.last_xyz_global)))]))[:3].T)
        # print("-----------")

        output = np.zeros((self.resolution[1], self.resolution[0], 2))
        output[(self.obj_uvs+0.5).astype(np.int32)[:, 1], (self.obj_uvs+0.5).astype(np.int32)[:, 0]] = (current_uv - self.obj_uvs)# [..., ::-1]
        # output[:, :, 0] = 0.5 + (output[:, :, 0] / self.resolution[0] / 2)
        self.flows.append(output)
        self.update_object_attr()

# trying to support multiple objects
class FlowManagerv2():
    def __init__(self, env, object_ids=[0], resolution=(320, 240), camera_name='corner', record_depths=False):
        self.flows = []
        self.env = env
        self.object_ids = object_ids
        self.resolution = resolution
        self.camera_name = camera_name
        self.record_depths = record_depths
        if record_depths:
            self.depths = {
                "before": [],
                "after": []
            }
        self.rel_transforms = []
        self.update_object_attr()

    def update_object_attr(self):
        xyz_global, uvd, seg_ids = get_obj_xyz_all(self.env, self.camera_name, self.resolution)
        self.obj_xyz_local = xyz_global_to_local_all(self.env, xyz_global, self.camera_name, seg_ids)
        if self.record_depths:
            self.last_depth = uvd
        self.obj_uvs, _ = xyz_to_uv(self.env, self.camera_name, xyz_global, self.resolution)
        self.last_pos = self.env.sim.data.geom_xpos[self.object_ids[0]].copy()
        self.last_rot = self.env.sim.data.geom_xmat[self.object_ids[0]].reshape(3, 3).copy()
        self.last_xyz_global = xyz_global.copy()
        self.seg_ids = seg_ids.copy()
    
    def append_flow(self):
        current_xyz_global = xyz_local_to_global_all(self.env, self.obj_xyz_local, self.camera_name, self.seg_ids)
        current_uv, current_d = xyz_to_uv(self.env, self.camera_name, current_xyz_global, self.resolution)
        if self.record_depths:
            self.depths["before"].append(self.last_depth)
            self.depths["after"].append(current_d)
            
        ### calculate extrinsic matrix
        last_ext = construct_extrinsic_matrix(self.last_pos, self.last_rot)
        current_ext = construct_extrinsic_matrix(self.env.sim.data.geom_xpos[self.object_ids[0]], self.env.sim.data.geom_xmat[self.object_ids[0]].reshape(3, 3))
        rel_ext = current_ext @ np.linalg.inv(last_ext)
        self.rel_transforms.append(rel_ext)

        # ### make sure that the gt transform works
        # print("-----------")
        # print(self.last_xyz_global[:5])
        # print(current_xyz_global[:5])
        # print((rel_ext @ np.concatenate([self.last_xyz_global.T, np.ones((1, len(self.last_xyz_global)))]))[:3].T[:5])
        # assert np.allclose(current_xyz_global, (rel_ext @ np.concatenate([self.last_xyz_global.T, np.ones((1, len(self.last_xyz_global)))]))[:3].T)
        # print("-----------")

        output = np.zeros((self.resolution[1], self.resolution[0], 2))
        output[(self.obj_uvs+0.5).astype(np.int32)[:, 1], (self.obj_uvs+0.5).astype(np.int32)[:, 0]] = (current_uv - self.obj_uvs)# [..., ::-1]
        # output[:, :, 0] = 0.5 + (output[:, :, 0] / self.resolution[0] / 2)
        self.flows.append(output)
        self.update_object_attr()



def collect_video(init_obs, env, policy, camera_name='corner3', resolution=(640, 480), flow_manager=None, sample_flow_ind=None):
    images = []
    depths = []
    episode_return = 0
    done = False
    obs = init_obs
    if camera_name is None:
        cameras = ["corner3", "corner", "corner2"]
        camera_name = np.random.choice(cameras)
    
    if flow_manager is not None:
        assert sample_flow_ind is not None

    image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution)
    images += [image]
    depths += [depth]

    dd = 10 ### collect a few more steps after done
    while dd:
        action = policy.get_action(obs)
        try:
            # print(env.step(action))
            obs, reward, done, info = env.step(np.float64(action))
            # print(done)
            done = info['success']
            dd -= done
            episode_return += reward
        except Exception as e:
            # raise e
            print(e)
            break
        if dd != 10 and not done:
            break
        image, depth = env.render(depth=True, offscreen=True, camera_name=camera_name, resolution=resolution)
        images += [image]
        depths += [depth]
        # print(len(images))
        if flow_manager is not None and len(images) in sample_flow_ind:
            flow_manager.append_flow()
        
    return images, depths, episode_return

def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])



    
    