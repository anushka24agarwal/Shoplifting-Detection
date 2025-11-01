import os
import json
import numpy as np
from glob import glob

def load_json_folder(json_folder):
    """Load PoseLift JSONs into a dict"""
    all_sequences = {}
    json_files = glob(os.path.join(json_folder, "*.json"))
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        for pid, frames in data.items():
            pid = str(pid)
            if pid not in all_sequences:
                all_sequences[pid] = {}

            for fid, info in frames.items():
                # Convert list to dict if needed
                if isinstance(info, list):
                    keypoints = info
                elif isinstance(info, dict) and 'keypoints' in info:
                    keypoints = info['keypoints']
                else:
                    print(f"Skipping unexpected frame format for pid={pid}, fid={fid}: {type(info)}")
                    continue

                # Convert 2D -> 3D if needed
                if len(keypoints) == 34:  # 17 * 2
                    keypoints_3d = []
                    for i in range(0, len(keypoints), 2):
                        x = keypoints[i]
                        y = keypoints[i+1]
                        z = 0.0
                        keypoints_3d.extend([x, y, z])
                    all_sequences[pid][fid] = keypoints_3d
                else:
                    all_sequences[pid][fid] = keypoints
    return all_sequences



def build_sequences(data, window=32):
    """Sliding window sequences (num_sequences, window, 51)"""
    sequences = []
    for pid, frames in data.items():
        sorted_fids = sorted(frames.keys(), key=lambda x: int(x))
        kp_list = [frames[fid] for fid in sorted_fids]
        kp_arr = np.array(kp_list, dtype=np.float32)
        if len(kp_arr) >= window:
            for i in range(0, len(kp_arr)-window+1):
                sequences.append(kp_arr[i:i+window])
    if len(sequences) == 0:
        raise ValueError("No valid sequences built. Check JSON structure or window size.")
    return np.stack(sequences)

# ----------------------------
# Example usage
# ----------------------------

video_json = "pose/video2_poselift.json"
video_data = load_json_folder(os.path.dirname(video_json))
video_sequences = build_sequences(video_data, window=32)
save_path = "pose/video2_poselift_3d.json"
with open(save_path, "w") as f:
    json.dump(video_data, f)
print(f"Saved 51-dim PoseLift JSON to {save_path}")



print("Video sequences shape (num_sequences, window, 51):", video_sequences.shape)

