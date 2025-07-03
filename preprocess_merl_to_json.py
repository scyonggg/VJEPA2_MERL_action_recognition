import os
import json
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# Action mapping (index to label)
ACTION_LABELS = [
    "Reach To Shelf",
    "Retract From Shelf",
    "Hand In Shelf",
    "Inspect Product",
    "Inspect Shelf"
]

# Data split definitions
TRAIN_SUBJECTS = set(range(1, 21))      # 1-20
VAL_SUBJECTS = set(range(21, 27))       # 21-26
TEST_SUBJECTS = set(range(27, 42))      # 27-41

LABELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Labels_MERL_Shopping_Dataset'))
VIDEOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Videos_MERL_Shopping_Dataset'))
OUTPUT_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), 'merl_data_new.json'))

# Helper to parse subject/session from filename
def parse_subject_session(filename):
    # e.g., 10_1_label.mat -> (10, 1)
    base = os.path.basename(filename)
    parts = base.split('_')
    subject = int(parts[0])
    session = int(parts[1])
    return subject, session

def get_split(subject):
    if subject in TRAIN_SUBJECTS:
        return 'train'
    elif subject in VAL_SUBJECTS:
        return 'validation'
    elif subject in TEST_SUBJECTS:
        return 'test'
    else:
        return None

def main():
    data = {'train': [], 'validation': [], 'test': []}
    label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('_label.mat')]
    for label_file in tqdm(label_files, desc='Processing label files'):
        subject, session = parse_subject_session(label_file)
        split = get_split(subject)
        if split is None:
            continue
        video_file = f"{subject}_{session}_crop.mp4"
        video_path = os.path.join(VIDEOS_DIR, video_file)
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue
        mat_path = os.path.join(LABELS_DIR, label_file)
        mat = loadmat(mat_path)
        # tlabs is a cell array (object array in numpy)
        tlabs = mat['tlabs'].squeeze()
        annotations = []
        for i, action_segments in enumerate(tlabs):
            # Each action_segments is a Kx2 array (K segments for this action)
            if action_segments.size == 0:
                continue
            segments = np.atleast_2d(action_segments)
            for seg in segments:
                start, end = int(seg[0]), int(seg[1])
                annotations.append({
                    'label': ACTION_LABELS[i],
                    'segment': [start, end]
                })
        data[split].append({
            'video_path': video_path,
            'annotations': annotations
        })
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved preprocessed data to {OUTPUT_JSON}")

if __name__ == '__main__':
    main() 