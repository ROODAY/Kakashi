import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import os
import errno
import argparse

from common.camera import *
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
from common.skeleton import Skeleton

def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def render_animation(poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6):
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index+1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        #ax.set_aspect('equal')
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
    all_frames = np.zeros((np.squeeze(np.array(poses), axis=0).shape[0], viewport[1], viewport[0]), dtype='uint8')
    
    if downsample > 1:
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        if not initialized:
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

parser = argparse.ArgumentParser(description='Generate Animation')
parser.add_argument('--viz-input', type=str, metavar='PATH', help='input file path')
parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
args = parser.parse_args()
assert args.viz_input is not None and args.viz_output is not None, "Input and output must be given!"

print('=> Rendering...')
prediction = np.load(args.viz_input)
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32) # Example value taken from h36m
prediction = camera_to_world(prediction, R=rot, t=0) # Invert camera transformation
prediction[:, :, 2] -= np.min(prediction[:, :, 2]) # We don't have the trajectory, but at least we can rebase the height

anim_output = {'Inference': prediction}
skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
render_animation(anim_output, skeleton, 24, 3000, 70.0, args.viz_output, viewport=(1000, 1002))
