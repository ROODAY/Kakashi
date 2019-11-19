from pathlib import Path
import subprocess
import argparse

def main(args):
  # make sure video directory exists
  download_dir = Path(Path.cwd(), 'data/', args.label)
  if not download_dir.exists():
    raise NotADirectoryError('{} does not exist!'.format(str(download_dir)))

  # get paths and cuts
  video_paths = sorted(list(Path(download_dir).rglob('*.mp4')))
  cut_file = Path(Path.cwd(), 'cuts/{}.txt'.format(args.label))
  cuts = [tuple(line.rstrip('\n').split()) for line in open(cut_file)]

  # cut each video, delete original, and rename cut
  for video_path, cut in zip(video_paths, cuts):
    start = cut[0]
    end = cut[1]
    cut_path = Path(video_path.parent, '{}.cut.mp4'.format(video_path.stem))
    command = 'ffmpeg -hwaccel nvdec -i {} -ss {} -to {} -c copy {}'.format(video_path, start, end, cut_path)
    subprocess.call(command, shell=True)
    cut_path.replace(video_path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Cut videos in a given data directory')
  parser.add_argument('label', type=str,
                      help='Label for the dataset (e.x. Popping)')
  args = parser.parse_args()
  main(args)
