# BoT-SORT
Modernized version of the [BoT-SORT](https://github.com/NirAharon/BoT-SORT) tracker published by Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky in [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651).

<p align="center"><img src="assets/Results_Bubbles.png"/></p>

## Highlights 
- Improved Kalman filter state vector over SORT tracking.
- Multi-class support.

## To Do
- [ ] Create OpenCV VideoStab GMC python binding or <u>write Python version<u>.
- [ ] Deployment code.
- [ ] Camera motion compensation modernization.
- [ ] Re-identification modernization.

## Abstract
The goal of multi-object tracking (MOT) is detecting and tracking all the objects in a scene, while keeping a unique identifier for each object. BoT-SORT is a robust state-of-the-art tracker, which can combine the advantages of motion and appearance information, along with camera-motion compensation, and a more accurate Kalman filter state vector. 

## Tracking performance
### Results on MOT17 challenge test set
| Tracker       |  MOTA |  IDF1  |  HOTA  |
|:--------------|:-------:|:------:|:------:|
| BoT-SORT      |  80.6   |  79.5  |  64.6  |
| BoT-SORT-ReID |  80.5   |  80.2  |  65.0  |

### Results on MOT20 challenge test set
| Tracker       | MOTA   | IDF1 | HOTA |
|:--------------|:-------:|:------:|:------:|
|BoT-SORT       | 77.7   | 76.3 | 62.6 | 
|BoT-SORT-ReID  | 77.8   | 77.5 | 63.3 | 


## Installation
### Pip Install from GitHub
Install from [GitHub](https://github.com/natsunoyuki/BoT-SORT) using `pip`

```bash
pip install git+https://github.com/natsunoyuki/BoT-SORT
```

### Local Install (Developer Mode)
Clone the repository from [GitHub](https://github.com/natsunoyuki/BoT-SORT) and install locally in developer mode to implement your customizations.

```bash
git clone https://github.com/natsunoyuki/BoT-SORT
cd BoT-SORT
pip install -e .
```

To install the package with tests, specify the corresponding install options.
```bash
git clone https://github.com/natsunoyuki/BoT-SORT
cd BoT-SORT
pip install -e ".[test]"
```

## Data Preparation
To do.


## Usage
```python
import cv2
from bot_sort import BoTSORT

# Initialize object detector.
detector = ...

# Initialize BoTSORT tracker.
tracker = BoTSORT(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30,
)

vid_cap = cv2.VideoCapture(...)
success = True

while success:
    success, bgr = vid_cap.read()
    if not success:
        break
    
    # Object detection to get the bounding boxes, labels and scores.
    # dets in this case is a dict with keys "bboxes", "labels" and "scores".
    dets = detector(bgr)

    # Update tracker.
    tracked_objects = tracker.update(
        dets["bboxes"], dets["labels"], dets["scores"],
    )

    # Get tracking ids.
    ids = [o.track_id for o in tracked_objects]
    ...

vid_cap.release()
```

### BoT-SORT Parameters
```
track_high_thresh=0.6: High detection score threshold. Detections with high scores will be automatically kept.
track_low_thresh=0.1: Minimum detection score. All detections with low scores will be dropped.
new_track_thresh=0.7: Detection score threshold to initiate new unconfirmed (tracks with only one beginning frame) tracks.
match_thresh=0.8: Minimum matching score threshold to match tracked bounding boxes.
track_buffer=30: Number of buffer frames. Typically set to the same as frame_rate.
frame_rate=30: Frame rate. Typically set to the same as track_buffer.
```

## Note About Camera Motion Compensation Module
Our camera motion compensation module is based on the OpenCV contrib C++ version of VideoStab Global Motion Estimation, 
which currently does not have a Python version. <br>
Motion files can be generated using the C++ project called 'VideoCameraCorrection' in the GMC folder. <br> 
The generated files can be used from the tracker. <br>

In addition, python-based motion estimation techniques are available and can be chosen by passing <br> 
'--cmc-method' <files | orb | ecc> to demo.py or track.py. 

## Acknowledgements
This repository is a modernized version of the original [BoT-SORT](https://github.com/NirAharon/BoT-SORT) repository. This modernized implementation is neither supported or funded by, nor affiliated with the original authors of [BoT-SORT](https://github.com/NirAharon/BoT-SORT).

The original paper by Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky should be cited as follows.
```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```

A large part of the code, ideas and results are borrowed from 
[ByteTrack](https://github.com/ifzhang/ByteTrack), 
[StrongSORT](https://github.com/dyhBUPT/StrongSORT),
[FastReID](https://github.com/JDAI-CV/fast-reid),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[YOLOv7](https://github.com/wongkinyiu/yolov7). 
Thanks for their excellent work!
