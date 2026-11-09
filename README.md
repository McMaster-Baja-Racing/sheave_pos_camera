# Sheave Position Camera

Track primary and secondary CVT sheave gaps in a video, export measurements to CSV, and plot the gaps over time.

## Requirements

- Python 3
- OpenCV, NumPy, and Matplotlib
- A desktop environment for the OpenCV windows and video file picker

Install the Python packages from the project root:

```powershell
python -m pip install opencv-python numpy matplotlib
```

On Windows, Tkinter is included with the standard Python installer and provides the video file picker.

## Run the Tracker

From the project root, run:

```powershell
python .\py\opencv_prim_sec.py
```

Choose a video in the file picker. You can also pass a video path to skip the picker:

```powershell
python .\py\opencv_prim_sec.py .\videos\GH020851.MP4
```

On the first frame, draw and accept each short line segment in this order:

1. Primary first sheave boundary
2. Primary second sheave boundary
3. Secondary first sheave boundary
4. Secondary second sheave boundary
5. One belt edge
6. The opposite belt edge, parallel to the first

Drag each line along its edge, then press `Enter` or `Space` to accept. Press `R` to clear and redraw the current line. Use short, locally straight segments where possible. The gap between the belt-edge lines is the 0.905-inch reference; draw them along a visible section of the belt, not across the belt or along its full length.

The initial sheave gaps are assumed to be 1.700 inches for the primary and 0.905 inches for the secondary. The primary gap is constrained to 0.905-1.700 inches; the secondary gap can increase from 0.905 inches. The belt reference and known starting gaps provide local scale calibration, but do not fully correct fisheye lens distortion.

The tracker displays the measured gaps while the video plays. Press `Q` or `Esc` to stop. A CSV is written to `results` when playback ends, with frame index, timestamp in seconds, primary gap in inches, and secondary gap in inches.

## Generate Plots

From the project root, run:

```powershell
python .\results\plot_travel.py
```

The plotting script creates one PNG per CSV in `results\plots`. Each plot shows primary and secondary gap values in inches against time calculated at 60 frames per second. It plots the recorded gaps directly, not the change from the first frame.
"# sheave_pos_camera" 
