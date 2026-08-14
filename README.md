# 🚧 Cement Bag Unload Counter

An intelligent Computer Vision pipeline designed to automatically track and count cement bags as they are unloaded from trucks in CCTV footage. The system uses state-of-the-art YOLO object detection combined with an interactive user interface to guarantee 100% counting accuracy, regardless of the camera's angle or resolution.

---

## 📖 Project Overview

When setting up CCTV cameras on a construction or unloading site, the camera angle is never perfectly identical. Because of this, hardcoded counting zones often break. 

This project solves that problem by allowing the user to **interactively draw the counting finish line** directly onto the video. Once the line is drawn, the system dynamically calculates the required math, builds a robust dual-line state machine in the background, and counts the workers crossing the path. It saves your drawn line so you only have to configure it once per camera!

---

## 📂 The Three Core Scripts

The project is broken down into three simple, modular Python scripts to make it easy to maintain and understand:

### 1. `1_extract_frames.py`
**Purpose:** Prepares raw data for AI training.
When you feed a raw CCTV video into this script, it slices the video up and saves exactly 1 frame per second as a standard image (JPEG). You can then upload these images to an annotation tool (like Roboflow) to draw boxes around the workers/cement bags and teach the AI what they look like.

### 2. `2_train_model.py`
**Purpose:** Teaches the AI to recognize the cement bags.
Once you have your dataset ready, this script fires up the YOLO AI engine. It is specifically optimized to utilize Apple Silicon GPU acceleration (`mps`) to rapidly train the AI on your custom images. It spits out a `best.pt` file, which is the "brain" of your new, fully-trained model.

### 3. `3_main_counter.py`
**Purpose:** The main counting application.
This is the workhorse of the project. It loads your trained AI model and runs it against a CCTV video. It handles the interactive UI (letting you draw the line), executes the object tracking algorithms, counts the bags passing the line, and generates a final resulting video with a sleek dashboard overlay.

---

## ⚙️ Detailed Workflow

1. **Data Collection:** You record CCTV footage of workers unloading cement bags.
2. **Extraction:** You run script `1` to break that video into images.
3. **Training:** You label the images and run script `2` to teach the AI what to look for. (If you don't do this, script 3 will safely fall back to the default YOLO model).
4. **Calibration (One-Time):** You run script `3`. A window pops up. You click-and-drag your mouse across the workers' path to draw the "finish line". The system saves this line to a configuration file.
5. **Processing:** The system tracks every single worker. As a worker approaches your line, they are mathematically tracked. Once they fully cross it, the top-left dashboard ticks up the `COUNTED` number.
6. **Output:** A final `.mp4` video is saved in the `output/` directory showing the full process visually.

---

## 🚀 How to Implement This Project (Step-by-Step for Beginners)

If you are setting this up on a brand new computer, follow these simple steps to get it running!

### Step 1: Install Prerequisites
You will need **Python** installed on your computer. If you are on a Mac, you can download it from python.org.

### Step 2: Open the Terminal
Open your computer's `Terminal` application and navigate to the folder where you downloaded this project.
```bash
cd path/to/CV_cemet_project
```

### Step 3: Create a Virtual Environment (Safe Space)
A virtual environment is like a sandbox. It ensures the tools you install for this project don't mess up your computer's main settings.
```bash
python3 -m venv .venv
```
Now, "activate" the sandbox:
* **On Mac/Linux:** `source .venv/bin/activate`
* **On Windows:** `.venv\Scripts\activate`

### Step 4: Install Dependencies
With your sandbox active, install the required AI libraries (like YOLO, OpenCV, and Supervision) that the scripts need to function:
```bash
pip install ultralytics opencv-python supervision numpy
```

### Step 5: Put Your Video in the Folder
Place the CCTV video you want to process into the `input/` directory inside the project folder. For example, let's say your video is named `my_video.mp4`.

### Step 6: Run the Counter!
Type the following command into your terminal:
```bash
python 3_main_counter.py --input input/my_video.mp4
```
1. A window will pop up showing the first frame of your video.
2. **Click and drag** your mouse to draw a line across the path the workers walk. 
3. **Press the `ENTER` key** on your keyboard to lock it in.
4. Let the script run! It will process the video and save the final result in the `output/` folder.

**Note:** If you ever want to redraw the line because you made a mistake, simply add `--recalibrate` to the end of your command!
```bash
python 3_main_counter.py --input input/my_video.mp4 --recalibrate
```
