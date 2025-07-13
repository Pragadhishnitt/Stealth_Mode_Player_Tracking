# Stealth_Mode_Player_Tracking_ReIdentification

## Description

This project showcases three different approaches to multi-object tracking using **YOLOv8** for detection:
1. **Basic IOU + Color Histogram Tracking**
2. **DeepSORT Tracker via `deep_sort_realtime`**
3. **Custom DeepSORT Tracker using `mars-small128.pb`**

Each section includes setup instructions for **Google Colab** and running locally on **Windows, macOS, or Ubuntu**.

---

## Table of Contents

1. [Basic IOU + Color Histogram](#1-basic-iou--color-histogram)
2. [DeepSORT using `deep_sort_realtime`](#2-deepsort-using-deep_sort_realtime)
3. [Custom DeepSORT Tracker](#3-custom-deepsort-tracker)

---

**I ran all the codes in colab and haven't tried running it in localhost as I didn't have a GPU support but will mention how to run it in local pc as well**

## 1. Basic IOU + Color Histogram

### Description

A simple tracker that uses IOU-based matching and color histogram similarity to maintain consistent IDs across frames.

### Usage in Colab 

Connect to any GPUs by changing the runtime before uploading the files

1. Upload `basic_tracker.ipynb` and your input video (`15sec_input_720p.mp4`) along with the trained yolo weights (`best.pt`).
2. Run all cells


---

## 2. DeepSORT using `deep_sort_realtime`

### Description

Uses the `deep_sort_realtime` library for high-performance tracking with YOLOv8.

### Usage in Colab

1. Upload:
    - Your YOLOv8 model (`best.pt`)
    - Your input video (`15sec_input_720p.mp4`)
    - Upload `deepsort.ipynb`
2. Run the script by clicking run all

---

## 3. Custom DeepSORT Tracker (Takes much longer time to run when compared to previous two approaches)

### Description

Implements DeepSORT from source using:
- `mars-small128.pb` (appearance model)
- Custom distance metric
- Classic DeepSORT tracking logic

### Usage in Colab

1. Upload:
    - `tracker.py`
    - all files from `model_data` directory
    - `custom_deepsort.ipynb`
    - Input video `15sec_input_720p.mp4`
    - `best.pt` -> Model weights

2. Run the script by clicking run all

---

## Notes
- Always recommended to run it using TPU or T4-GPU in colab
- YOLOv8 model (`best.pt`) has been trained already for our use case.
- Recommended confidence threshold: **0.85–0.9** for deepsort.ipynb and ****.
- Avoid tracking classes not required (`cls_id != 2`, etc.).
- Ensure consistent frame sizes when using `DeepSORT`.

---
### Local 
**Git LFS Support**
This repo uses Git Large File Storage (LFS) to handle large files like model weights and videos (e.g., best.pt, *.mp4).
**Setting up Git LFS**
Before cloning or pushing to this repo, ensure Git LFS is installed and initialized:

Ubuntu/macOS:
```bash
sudo apt install git-lfs      # Or: brew install git-lfs
git lfs install
```

Windows:
    Download and install from: https://git-lfs.github.com/
    Then run: git lfs install (in Git Bash)
    
1. git clone <https://github.com/Pragadhishnitt/Stealth_Mode_Player_Tracking.git>
2. cd stealth_mind_tracker_re_id
3. Optional Step :  Better to run using venv

**a) Ubuntu/MacOs**

```bash
python3 -m venv venv
source venv/bin/activate
```

**b) Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

4. 

**(can comment the ones not needed if running specific program)**

```bash 
pip install -r requirements.txt
```
If any mismatch download the required version suggested in the terminal ( The versions mentioned here are as per the versions used in Colab )
**`requirements_2.txt` file contains specified without versions for that aspect**

5. Run : 

**Ensure to change target output directories before running the program**

```bash
python basic_tracker.py     #(for running basic_iou_histogram_tracker)

python deepsort.py          #(for running using pre built deepsort tracker)

For Custom Tracker :
Step 1:
git clone https://github.com/nwojke/deep_sort.git #(Clone deepsort repository first)

Step 2:
python custom_deepsort.py   #(for running using custom deepsort tracker)
```

   **(Replace python by python3 when run in MacOs/Ubuntu)**

## Folder Structure

<img width="685" height="250" alt="image" src="https://github.com/user-attachments/assets/de39eb93-b05c-4ee2-8f7f-182a1d91a9ca" />

## Output

Each script saves an annotated output video with bounding boxes and IDs for each tracked object.

---

## Future Improvements

- Integrate visual analytics (count per class, trajectory heatmaps)
- Add support for ReID models like OSNet
- Switch to ByteTrack or OCSORT for faster tracking
