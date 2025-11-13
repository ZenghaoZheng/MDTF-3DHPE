import copy
import pickle
import torch 
import numpy as np
import random
from itertools import groupby
class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/data/augmentation.py#L10
    """
    def __init__(self, args):
        self.d2c_params = read_pkl(args.d2c_params_path)
        self.mean_noise = torch.load(args.mean_noise_path)
        self.std_noise = torch.load(args.std_noise_path)
        self.mask_ratio = args.mask_ratio
        self.mask_T_ratio = args.mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        f = a/(dis+a)+b*dis
        shift = torch.randn(*dis.shape)*s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift

    # def add_noise(self, motion_2d):
    #     a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
    #     if "uniform_range" in self.noise.keys():
    #         uniform_range = self.noise["uniform_range"]
    #     else:
    #         uniform_range = 0.06
    #     motion_2d = motion_2d[:,:,:,:2]
    #     batch_size = motion_2d.shape[0]
    #     num_frames = motion_2d.shape[1]
    #     num_joints = motion_2d.shape[2]
    #     mean = self.noise['mean'].float()
    #     std = self.noise['std'].float()
    #     weight = self.noise['weight'][:,None].float()
    #     sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
    #     gaussian_sample = (torch.randn(batch_size, self.num_Kframes, num_joints, 2) * std + mean)
    #     uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2))-0.5) * uniform_range
    #     noise_mean = 0
    #     delta_noise = torch.randn(num_frames, num_joints, 2) * self.noise_std + noise_mean
    #     # if torch.cuda.is_available():
    #     mean = mean.to(motion_2d.device)
    #     std = std.to(motion_2d.device)
    #     weight = weight.to(motion_2d.device)
    #     gaussian_sample = gaussian_sample.to(motion_2d.device)
    #     uniform_sample = uniform_sample.to(motion_2d.device)
    #     sel = sel.to(motion_2d.device)
    #     delta_noise = delta_noise.to(motion_2d.device)
    #
    #     delta = gaussian_sample*(sel<weight) + uniform_sample*(sel>=weight)
    #     delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2], mode='trilinear', align_corners=True)[:,0]
    #     delta_final = delta_expand + delta_noise
    #     motion_2d = motion_2d + delta_final
    #     dx = delta_final[:,:,:,0]
    #     dy = delta_final[:,:,:,1]
    #     dis2 = dx*dx+dy*dy
    #     dis = torch.sqrt(dis2)
    #     conf = self.dis2conf(dis, a, b, m, s).clip(0,1).reshape([batch_size, num_frames, num_joints, -1])
    #     return torch.cat((motion_2d, conf), dim=3)
    def add_noise(self, motion_2d):
        motion_2d = motion_2d[:, :, :, :2]

        mean = self.mean_noise.to(motion_2d.device)
        std = self.std_noise.to(motion_2d.device)

        batch_size = motion_2d.shape[0]
        num_frames = motion_2d.shape[1]
        num_joints = motion_2d.shape[2]

        uniform_range = 0.06
        gaussian_sample = torch.randn((batch_size, self.num_Kframes, num_joints, 2), device=motion_2d.device) * std + mean
        uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2), device=motion_2d.device) - 0.5) * uniform_range

        weight = 0.5
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1), device=motion_2d.device)
        delta = gaussian_sample * (sel < weight) + uniform_sample * (sel >= weight)

        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2],
                                                       mode='trilinear', align_corners=True)[:, 0]

        noise_mean = 0
        noise_std = 0.002
        delta_noise = torch.randn(num_frames, num_joints, 2, device=motion_2d.device) * noise_std + noise_mean

        delta_final = delta_expand + delta_noise
        motion_2d = motion_2d + delta_final

        # 计算距离并生成置信度
        dx = delta_final[:, :, :, 0]
        dy = delta_final[:, :, :, 1]
        dis2 = dx * dx + dy * dy
        dis = torch.sqrt(dis2)

        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        conf = self.dis2conf(dis, a, b, m, s).clip(0, 1).reshape([batch_size, num_frames, num_joints, -1])
        return torch.cat((motion_2d, conf), dim=3)

    def generate_mask_T(self, N, T, device):
        mask_types = ['future', 'past', 'middle']

        mask_T = torch.ones((N, T, 1, 1), dtype=torch.bool, device=device)
        mask_length = int(T * self.mask_T_ratio)

        for i in range(N):
            mask_type = random.choice(mask_types)

            if mask_type == 'future':
                start = T - mask_length
                mask_T[i, start:, 0, 0] = 0

            elif mask_type == 'past':
                end = mask_length
                mask_T[i, :end, 0, 0] = 0

            elif mask_type == 'middle':
                center = T // 2
                half_mask_length = mask_length // 2
                if self.mask_T_ratio >= 1/3:
                    start = center - half_mask_length
                    end = center + half_mask_length + (mask_length % 2)
                else:
                    start_idx = torch.randint(low=mask_length, high=T - mask_length - mask_length + 1, size=(1,)).item()
                    start = start_idx
                    end = start_idx + mask_length

                mask_T[i, start:end, 0, 0] = 0

        return mask_T

    # def add_mask(self, motion2d, motion3d, mean, std, fill_mask):
    #     '''
    #     Add mask to data
    #     x.shape = (batch_size, frame, num_joints, 3)
    #     y.shape = (batch_size, frame, num_joints, 2)
    #     '''
    #     batch,f,j,c = motion2d.shape
    #     mask = []
    #     # mask_T = []
    #     for y in motion3d:
    #         diffs = torch.norm(torch.diff(y, dim=0), dim=-1)
    #         groups = {
    #             '右腿': [1, 2, 3],
    #             '左腿': [4, 5, 6],
    #             '脊柱': [0, 7, 8, 9, 10],
    #             '右胳膊': [14, 15, 16],
    #             '左胳膊': [11, 12, 13]
    #         }
    #         # significant_frames = {}
    #         mask_s = torch.ones(f, j, c)
    #         for name, indices in groups.items():
    #             overall_movement = torch.sum(diffs[:, indices], dim=1)
    #             threshold = torch.mean(overall_movement) * 1
    #             significant_indices = torch.where(overall_movement > threshold)[0] + 1      #高幅度动作范围
    #
    #             num_to_mask = int(np.ceil(len(significant_indices) * self.mask_ratio))
    #             selective = torch.randperm(significant_indices.size(0))[:num_to_mask]
    #             indices_to_mask = significant_indices[selective]
    #             for frame_idx in indices_to_mask:
    #                  mask_s[frame_idx, indices, :] = 0
    #
    #         mask.append(mask_s)
    #             # continuous_segments = []
    #             # for k, g in groupby(enumerate(significant_indices), lambda ix: ix[0] - ix[1]):
    #             #     segment = list(map(lambda ix: ix[1], g))
    #             #     continuous_segments.append(segment)
    #
    #             # significant_frames[name] = continuous_segments
    #             # significant_frames[name] = significant_indices
    #
    #         # mask_s = torch.ones(f,j,c)
    #         # for bodypart, frames in significant_frames.items():
    #         #     num_to_mask = int(np.ceil(len(frames) * self.mask_ratio))
    #         #     indices_to_mask = random.sample(frames.tolist(), num_to_mask)
    #         #     for frame_idx in indices_to_mask:
    #         #         mask_s[frame_idx, groups[bodypart], :] = 0
    #             # for segment in frames:
    #             #     num_to_mask = int(np.ceil(len(segment) * self.mask_ratio))
    #             #     indices_to_mask = random.sample(segment, num_to_mask)
    #             #     for frame_idx in indices_to_mask:
    #             #         mask_s[frame_idx, groups[bodypart], :] = 0
    #         # mask.append(mask_s.tolist())
    #
    #         # overall_movement = torch.sum(diffs, dim=1)
    #         # threshold = torch.mean(overall_movement) * 1
    #         # significant_indices = torch.where(overall_movement > threshold)[0] + 1
    #         # # continuous_segments = []
    #         # # for k, g in groupby(enumerate(significant_indices), lambda ix: ix[0] - ix[1]):
    #         # #     segment = list(map(lambda ix: ix[1], g))
    #         # #     continuous_segments.append(segment)
    #         #
    #         # mask_t = torch.ones(f,j,c)
    #         # # for segment in continuous_segments:
    #         # #     num_to_mask = int(np.ceil(len(segment) * self.mask_T_ratio))
    #         # #     indices_to_mask = random.sample(segment, num_to_mask)
    #         # #     for frame_idx in indices_to_mask:
    #         # #         mask_t[frame_idx, :, :] = 0
    #         # num_to_mask = int(np.ceil(len(significant_indices) * self.mask_T_ratio))
    #         # selective = torch.randperm(significant_indices.size(0))[:num_to_mask]
    #         # indices_to_mask = significant_indices[selective]
    #         # mask_t[indices_to_mask, :, :] = 0
    #         # mask_T.append(mask_t)
    #
    #     mask = torch.stack(mask, dim=0).to(motion2d.device)
    #     # mask_T = torch.stack(mask_T, dim=0).to(motion2d.device)
    #     mask_T = torch.rand(batch,f,1,1, dtype=motion2d.dtype, device=motion2d.device) > self.mask_T_ratio
    #     x = motion2d * mask * mask_T
    #
    #
    #     if fill_mask:
    #         noise = torch.randn((batch, f, j, 2), dtype=x.dtype, device=x.device)
    #         noise = noise * std + mean
    #
    #         dx = noise[:, :, :, 0]
    #         dy = noise[:, :, :, 1]
    #         dis2 = dx * dx + dy * dy
    #         dis = torch.sqrt(dis2)
    #         a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
    #         conf = self.dis2conf(dis, a, b, m, s).clip(0, 1).reshape([batch, f, j, -1])
    #         noise_conf = torch.cat((noise, conf), dim=3)
    #
    #         masked_x = x + (~mask_T) * noise_conf
    #         return masked_x
    #     else:
    #         return x

    def add_mask(self, x, mean, std, fill_mask):
        N, T, J, C = x.shape
        mask = torch.rand(N, T, J, 1, dtype=x.dtype, device=x.device) > self.mask_ratio
        mask_T = self.generate_mask_T(N, T, x.device)
        x = x * mask * mask_T

        if fill_mask:
            noise = torch.randn((N, T, J, 2), dtype=x.dtype, device=x.device)
            noise = noise * std + mean

            dx = noise[:, :, :, 0]
            dy = noise[:, :, :, 1]
            dis2 = dx * dx + dy * dy
            dis = torch.sqrt(dis2)
            a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
            conf = self.dis2conf(dis, a, b, m, s).clip(0, 1).reshape([N, T, J, -1])
            noise_conf = torch.cat((noise, conf), dim=3)

            masked_x = x + (~mask_T) * noise_conf
            return masked_x
        else:
            return x


    def augment2D(self, motion_2d, motion_3d, mask=False, noise=False, fill_mask=False):
        mean = motion_2d[:, :, :, :2].mean(dim=(0, 1), keepdim=True)
        std = motion_2d[:, :, :, :2].std(dim=(0, 1), keepdim=True)
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d, mean, std, fill_mask)
        return motion_2d


class Augmenter3D(object):
    """
        Make 3D augmentations when dataloaders get items. NumPy single motion version.
    """

    def __init__(self, args):
        self.flip = args.flip
        if hasattr(args, "scale_range_pretrain"):
            self.scale_range_pretrain = args.scale_range_pretrain
        else:
            self.scale_range_pretrain = None

    def augment3D(self, motion_3d):
        if self.scale_range_pretrain:
            motion_3d = crop_scale_3d(motion_3d, self.scale_range_pretrain)
        if self.flip and random.random() > 0.5:
            motion_3d = flip_data(motion_3d)
        return motion_3d


def linepose(x, joints):
    # x.shape: b f j c
    b, f, j, c = x.shape
    lambd = torch.rand((b, f, 5, 1), device=x.device)
    y = torch.zeros((b, f, joints, c), device=x.device)

    y[:, :, 0] = x[:, :, 0]
    y[:, :, (2, 4, 6)] = x[:, :, (1, 2, 3)]
    y[:, :, (8, 10, 12)] = x[:, :, (4, 5, 6)]
    y[:, :, (14, 16, 18, 20)] = x[:, :, (7, 8, 9, 10)]
    y[:, :, (22, 24, 26)] = x[:, :, (11, 12, 13)]
    y[:, :, (28, 30, 32)] = x[:, :, (14, 15, 16)]
    # 新的置信度假设按照线性比例
    y[:, :, (1, 3, 5)] = x[:, :, 0:3] + (x[:, :, 1:4] - x[:, :, 0:3]) * lambd[:, :, 0:1]
    y[:, :, (7, 9, 11)] = x[:, :, (0, 4, 5)] + (x[:, :, 4:7] - x[:, :, (0, 4, 5)]) * lambd[:, :, 1:2]
    y[:, :, (13, 15, 17, 19)] = x[:, :, (0, 7, 8, 9)] + (x[:, :, 7:11] - x[:, :, (0, 7, 8, 9)]) * lambd[:, :, 2:3]
    y[:, :, (21, 23, 25)] = x[:, :, (8, 11, 12)] + (x[:, :, 11:14] - x[:, :, (8, 11, 12)]) * lambd[:, :, 3:4]
    y[:, :, (27, 29, 31)] = x[:, :, (8, 14, 15)] + (x[:, :, 14:17] - x[:, :, (8, 14, 15)]) * lambd[:, :, 4:5]

    return y

def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
        Motion: [T, 17, 3]. (x, y, z)
        Normalize to [-1, 1]
        Z is relative to the first frame's root.
    '''
    result = copy.deepcopy(motion)
    result[:,:,2] = result[:,:,2] - result[0,0,2]
    xmin = np.min(motion[...,0])
    xmax = np.max(motion[...,0])
    ymin = np.min(motion[...,1])
    ymax = np.max(motion[...,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) / ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,2] = result[...,2] / scale
    result = (result - 0.5) * 2
    return result


def resample(ori_len, target_len, replay=False, randomness=True):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def split_clips(vid_list, n_frames, data_stride):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def posetrack2h36m(x):
    '''
        Input: x (T x V x C)

        PoseTrack keypoints = [ 'nose',
                                'head_bottom',
                                'head_top',
                                'left_ear',
                                'right_ear',
                                'left_shoulder',
                                'right_shoulder',
                                'left_elbow',
                                'right_elbow',
                                'left_wrist',
                                'right_wrist',
                                'left_hip',
                                'right_hip',
                                'left_knee',
                                'right_knee',
                                'left_ankle',
                                'right_ankle']
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = x[:,1,:]
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,2,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    y[:,0,2] = np.minimum(x[:,11,2], x[:,12,2])
    y[:,7,2] = np.minimum(y[:,0,2], y[:,8,2])
    return y


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def denormalize(pred, seq):
    out = pred.cpu().numpy()
    for idx in range(out.shape[0]):
        if seq[idx] in ['TS5', 'TS6']:
            res_w, res_h = 1920, 1080
        else:
            res_w, res_h = 2048, 2048
        out[idx, :, :, :2] = (out[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        out[idx, :, :, 2:] = out[idx, :, :, 2:] * res_w / 2
    out = out - out[..., 0:1, :]
    return torch.tensor(out).cuda()