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

1. Primary first sheave edge, away from the belt
2. Primary second sheave edge, away from the belt
3. Secondary first inner sheave edge, toward the belt
4. Secondary second inner sheave edge, toward the belt
5. One belt edge
6. The opposite belt edge, parallel to the first

Drag each short line along its edge, then press `Enter` or `Space` to accept. Press `R` to clear and redraw the current line. Use locally straight segments where possible. The gap between the belt-edge lines is the 0.905-inch reference; draw them along a visible section of the belt, not across the belt or along its full length.

The initial sheave gaps are assumed to be 1.700 inches for the primary and 0.905 inches for the secondary. The primary gap is constrained to 0.905-1.700 inches; the secondary gap can increase from 0.905 inches. The belt reference and known starting gaps provide local scale calibration, but do not fully correct fisheye lens distortion. Redrawing lines during playback resets their tracking references, not this original scale calibration.

Sheave lines are searched relative to their original first-frame positions, within 50 pixels along the edge normal. The tracker estimates each edge's signed contrast direction at startup, applies a 5x5 Gaussian blur before Sobel-gradient scoring, and uses a Huber line fit to reduce the effect of outlier edge samples. The line angle is limited to 10 degrees from its startup angle. Belt edges are tracked separately with optical flow.

During playback, press `Space` to pause and redraw all six lines on the current frame. Press `Enter` or `Space` to accept each line; playback resumes when the set is complete. The top status label shows elapsed video time calculated from frame index and video FPS, not wall-clock time. Press `Q` or `Esc` to stop. A CSV is written to `results` when playback ends, with frame index, timestamp in seconds, primary gap in inches, and secondary gap in inches.

## Generate Plots

From the project root, run:

```powershell
python .\results\plot_travel.py
```

The plotting script creates one PNG per CSV in `results\plots`. Each plot shows primary and secondary gap values in inches against time calculated at 60 frames per second. It plots the recorded gaps directly, not the change from the first frame.
"# sheave_pos_camera" 
