import os
import trackers

from trackers.bot_sort import BOTSORT
from trackers.utils import IterableSimpleNamespace, YAML

tracker_cfg_pathmap = {
    "botsort": "trackers/cfg/botsort.yaml"
}
tracker_classmap = {"botsort": BOTSORT}


def track_objects(datadir: str, tracker_method: str = "botsort") -> None:
    """
    - load detections from `detections/` directory.
    - load corresponding images from `images/` directory.
    - track detections in next frame.
    - render both detecctions and tracks in a image and save to visz_objdets_tracks/ dir. 
    """
    ts_file_path = os.path.join(datadir, "timestamps.txt")
    det_dir = os.path.join(datadir, "detections")
    images_dir = os.path.join(datadir, "images")
    if not os.path.exists(ts_file_path):
        raise FileNotFoundError(f"Timestamp file not found: {ts_file_path}")
    if not os.path.exists(det_dir):
        raise FileNotFoundError(f"Detections directory not found: {det_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    with open(ts_file_path, 'r') as file:
        timestamps = [line.strip() for line in file.readlines()]

    tracker_cfg_path = os.path.join(os.path.dirname(__file__), tracker_cfg_pathmap[tracker_method])
    tracker_cfg = IterableSimpleNamespace(**YAML.load(tracker_cfg_path))
    tracker = tracker_classmap[tracker_method]((args=tracker_cfg, frame_rate=30)

    # start processing
    for ts in timestamps:
        pass
