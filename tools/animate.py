import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp

from common.arguments import parse_args
from common.camera import *
from common.model import *
from common.loss import *
from common.visualization import render_animation
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D

def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def render_animation(poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Black background
    all_frames = np.zeros((poses['Reconstruction'].shape[0], viewport[1], viewport[0]), dtype='uint8')
    
    if downsample > 1:
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j-1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j-1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()

matplotlib.use('Agg')
args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Rendering...')
ground_truth = None
print('INFO: this action is unlabeled. Ground truth will not be rendered.')
prediction = np.load('/projectnb/dnn-motion/rooday/Kakashi/out/00001.keypoints.npy')
    
# Invert camera transformation
cam = dataset.cameras()[args.viz_subject][args.viz_camera]
# If the ground truth is not available, take the camera extrinsic params from a random subject.
# They are almost the same, and anyway, we only need this for visualization purposes.
for subject in dataset.cameras():
    if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
        rot = dataset.cameras()[subject][args.viz_camera]['orientation']
        break
prediction = camera_to_world(prediction, R=rot, t=0)
# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])

anim_output = {'Reconstruction': prediction}
render_animation(anim_output, dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                 limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size, viewport=(cam['res_w'], cam['res_h']))