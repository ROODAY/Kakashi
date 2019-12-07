# Kakashi
An LSTM RNN model to generate dance choreography (in the form of 3D pose matrices) from arbitrary music samples. Directed study project for Fall 2019 and Spring 2020, under Professor Margrit Betke at Boston University. Currently in progress.

## Installation/Setup
Kakashi requires Pytorch >= 1.1 and Python >= 3.7.3. Python dependencies can be installed with `pip install -r requirements.txt`. 

### Dataset Generation
To generate your own dataset, follow these instructions. If you'd like to use the Kakashi WOD dataset, it can be downloaded [here](https://tinyurl.com/kakashi-dataset) and these instructions can be ignored.

To generate the dataset, you'll need FFmpeg on your machine, as well as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) setup (follow its instructions for inference, which include setting up [Detectron](https://github.com/facebookresearch/Detectron)). Once these projects are setup, set the following environment variables:
```
$KAKASHI=/path/to/Kakashi/root
$VIDEOPOSE=/path/to/VideoPose3D/root
$DETECTRON=/path/to/Detectron/root
```

Then download a playlist of YouTube videos to use as ground truth using `python tools/download_playlist.py <DATASET_LABEL> --playlist_url <URL>`. If the videos need to be trimmed, create a file in the `cuts/` folder, named `<DATASET_LABEL>.txt`. Then run `python tools/cut_videos.py <DATASET_LABEL>`. Now, you can run the dataset generation script with `python tools/generate_dataset.py <DATASET_LABEL>`. This script will use VideoPose3D and Detectron to extract 3D pose keypoints from the videos, as well as use LibROSA to extract audio features. Be sure to check out the actual script as it has a variety of command line arguments. After this, the dataset is generated and ready to use!

## Training
After your dataset is generated, you can run the model simply with `python train.py <DATASET_LABEL>`. This file also takes a variety of arguments, including running deterministically, loading presaved iterators to save time, and using a config file for model paramters (check out the `config/` folder). This file will save the training, valid, and test iterators in the `its/` folder, as well as pretrained models in the `pre/` folder to save time in later runs.

##Inference
Once pretrained models are generated, you can infer output using `python infer.py path/to/model path/to/audio/file`. If you'd like to render the output of the inference, copy `tools/animate.py` to the `$VIDEOPOSE` folder, then run inference with the `--render` flag. You can also specify a path to save the rendered video with `--render_name`.