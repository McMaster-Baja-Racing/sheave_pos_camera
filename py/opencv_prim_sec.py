import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BELT_WIDTH_IN = 0.905
PRIMARY_MIN_GAP_IN = BELT_WIDTH_IN
PRIMARY_MAX_GAP_IN = 1.7
SECONDARY_MIN_GAP_IN = BELT_WIDTH_IN
MAX_LINE_ANGLE_CHANGE_DEG = 10
SHEAVE_SEARCH_RADIUS_PX = 50


class CVTSheaveTracker:
    def __init__(self, video_path=None):
        if video_path is None:
            video_path = self.select_video_file()

        if not video_path:
            print("No video selected.")
            raise SystemExit(0)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_property is not None:
            self.cap.set(orientation_property, 1)

        screen_w = 1920
        screen_h = 1080
        self.max_win_w = min(1200, int(screen_w * 0.6))
        self.max_win_h = min(800, int(screen_h * 0.6))
        self.win_w = self.max_win_w
        self.win_h = self.max_win_h
        
        # Track line-selection and optical-flow state.
        self.drawing = False
        self.line_start = None
        self.current_line = None
        self.line_endpoints = {}
        self.line_reference_endpoints = {}
        self.line_reference_angles = {}
        self.line_reference_polarities = {}
        self.line_points = {}
        self.previous_gray = None

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.belt_scale_factor = None
        self.primary_scale_factor = None
        self.secondary_scale_factor = None

    @staticmethod
    def select_video_file():
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        try:
            return filedialog.askopenfilename(
                parent=root,
                title="Select a CVT video",
                filetypes=(
                    ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.mpg *.mpeg"),
                    ("All files", "*.*"),
                ),
            )
        finally:
            root.destroy()

    def apply_clahe(self, gray_frame):
        """Normalizes dynamic lighting fluctuations and specular highlights."""
        return self.clahe.apply(gray_frame)

    def resize_frame(self, frame):
        frame_h, frame_w = frame.shape[:2]
        scale = min(
            self.max_win_w / frame_w,
            self.max_win_h / frame_h,
            1.0,
        )
        self.win_w = max(1, int(round(frame_w * scale)))
        self.win_h = max(1, int(round(frame_h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame, (self.win_w, self.win_h), interpolation=interpolation)

    @staticmethod
    def is_window_open(window_name):
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def mouse_callback(self, event, x, y, flags, param):
        """Capture a straight line drawn along a sheave boundary."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.line_start = (x, y)
            self.current_line = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            param['canvas'][:] = param['frame']
            cv2.line(param['canvas'], self.line_start, (x, y), param['color'], 3)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.line_start is not None:
                self.current_line = (self.line_start, (x, y))
                param['canvas'][:] = param['frame']
                cv2.line(param['canvas'], *self.current_line, param['color'], 3)

    def draw_sheave_selection(self, label, frame, color):
        """Let the user drag a line along one sheave boundary."""
        canvas = frame.copy()
        self.current_line = None
        
        win_name = "CVT Sheave Selection"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(
            win_name,
            self.mouse_callback,
            {'canvas': canvas, 'frame': frame, 'color': color},
        )

        while True:
            temp_display = canvas.copy()
            instruction = f"Drag a short line along {label}; ENTER accepts, R clears"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            (text_width, text_height), baseline = cv2.getTextSize(
                instruction, font, font_scale, 1
            )
            max_text_width = self.win_w - 40
            if text_width > max_text_width:
                font_scale *= max_text_width / text_width
                (text_width, text_height), baseline = cv2.getTextSize(
                    instruction, font, font_scale, 1
                )
            text_x = 20
            text_y = 8 + 10 + text_height

            # Top instruction overlay with a box sized to its text.
            cv2.rectangle(temp_display, (0, 0), (self.win_w, 45), (0, 0, 0), -1)
            cv2.rectangle(
                temp_display,
                (10, 8),
                (text_x + text_width + 10, text_y + baseline + 2),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                temp_display,
                instruction,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

            cv2.imshow(win_name, temp_display)
            key = cv2.waitKey(15) & 0xFF

            if not self.is_window_open(win_name):
                self.cap.release()
                cv2.destroyAllWindows()
                raise SystemExit(0)

            if key == ord('r'):
                canvas[:] = frame
                self.current_line = None
            elif key in [13, 32] and self.current_line is not None:
                break

        cv2.destroyWindow(win_name)
        if self.current_line is None:
            return None
        return self.current_line

    def initialize_line_points(self, gray, key):
        """Find trackable image features near a selected line."""
        mask = np.zeros(gray.shape, dtype=np.uint8)
        start, end = self.line_endpoints[key]
        cv2.line(mask, start, end, 255, thickness=9)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=5,
        )
        if points is None:
            self.line_points[key] = np.empty((0, 1, 2), dtype=np.float32)
        else:
            self.line_points[key] = points.astype(np.float32)

    def select_tracking_lines(self, frame, gray, drawn_targets, is_correction=False):
        for label, key, color in drawn_targets:
            if is_correction:
                label = f"Correct {label}"
            line = self.draw_sheave_selection(label, frame, color)
            if line is None:
                return False
            start, end = line
            if start[0] > end[0]:
                line = (end, start)
            self.line_endpoints[key] = line
            self.line_reference_endpoints[key] = line
            direction = np.asarray(line[1], dtype=np.float32) - line[0]
            self.line_reference_angles[key] = float(
                np.arctan2(direction[1], direction[0])
            )
            self.line_reference_polarities[key] = self.measure_line_polarity(
                gray,
                line,
            )
            self.initialize_line_points(gray, key)

        belt_angles = np.array([
            self.line_reference_angles['belt_edge_first'],
            self.line_reference_angles['belt_edge_second'],
        ])
        belt_angle = 0.5 * np.arctan2(
            np.sin(2 * belt_angles).mean(),
            np.cos(2 * belt_angles).mean(),
        )
        self.line_reference_angles['belt_edge_first'] = float(belt_angle)
        self.line_reference_angles['belt_edge_second'] = float(belt_angle)
        self.previous_gray = gray
        return True

    def track_line(self, gray, key):
        """Track features but move the selected line only perpendicular to itself."""
        points = self.line_points[key]
        preserve_direction = key in ('belt_edge_first', 'belt_edge_second')
        if preserve_direction and self.previous_gray is not None and len(points) > 0:
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            if next_points is not None and status is not None:
                valid = status.ravel() == 1
                if errors is not None:
                    valid &= errors.ravel() < 20
                if np.any(valid):
                    forward_points = next_points[valid]
                    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                        gray,
                        self.previous_gray,
                        forward_points,
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                    )
                    if backward_points is not None and backward_status is not None:
                        old_points = points.reshape(-1, 2)[valid]
                        new_points = forward_points.reshape(-1, 2)
                        backward_error = np.linalg.norm(
                            backward_points.reshape(-1, 2) - old_points,
                            axis=1,
                        )
                        consistent = (backward_status.ravel() == 1) & (backward_error < 1.5)
                        old_valid = old_points[consistent]
                        new_valid = new_points[consistent]

                        if len(new_valid) >= 3:
                            endpoints = np.asarray(self.line_endpoints[key], dtype=np.float32)
                            direction = endpoints[1] - endpoints[0]
                            direction /= max(np.linalg.norm(direction), 1e-6)
                            normal = np.array([-direction[1], direction[0]])
                            flows = new_valid - old_valid
                            normal_flows = flows @ normal
                            median_normal_flow = np.median(normal_flows)
                            deviations = np.abs(normal_flows - median_normal_flow)
                            flow_inliers = deviations <= max(
                                2.0,
                                float(np.median(deviations) * 2.5),
                            )
                            old_valid = old_valid[flow_inliers]
                            new_valid = new_valid[flow_inliers]
                            if len(new_valid) >= 3:
                                normal_flow = np.median((new_valid - old_valid) @ normal)
                                displacement = normal * normal_flow
                                endpoints += displacement
                                updated_line = tuple(
                                    tuple(np.round(point).astype(int)) for point in endpoints
                                )
                                updated_line = self.snap_line_to_edge(
                                    gray,
                                    updated_line,
                                    max_offset=4,
                                    preserve_direction=preserve_direction,
                                    preferred_polarity=self.line_reference_polarities.get(
                                        key, 0
                                    ),
                                )
                                updated_line = self.constrain_line_angle(
                                    updated_line,
                                    self.line_reference_angles[key],
                                )
                                line_inliers = self.points_near_line(
                                    new_valid,
                                    updated_line,
                                    6,
                                )
                                new_valid = new_valid[line_inliers]
                                if len(new_valid) >= 3:
                                    self.line_endpoints[key] = updated_line
                                    self.line_points[key] = new_valid.reshape(-1, 1, 2)
                                    return self.line_endpoints[key]

        if preserve_direction:
            reference_line = self.line_endpoints[key]
            max_offset = 6
        else:
            reference_line = self.line_reference_endpoints[key]
            max_offset = SHEAVE_SEARCH_RADIUS_PX
        line = self.snap_line_to_edge(
            gray,
            reference_line,
            max_offset=max_offset,
            preserve_direction=preserve_direction,
            preferred_polarity=self.line_reference_polarities.get(key, 0),
        )
        line = self.constrain_line_angle(line, self.line_reference_angles[key])
        self.line_endpoints[key] = line
        self.initialize_line_points(gray, key)
        return self.line_endpoints[key]

    @staticmethod
    def constrain_line_angle(
        line,
        reference_angle,
        max_change_degrees=MAX_LINE_ANGLE_CHANGE_DEG,
    ):
        endpoints = np.asarray(line, dtype=np.float32)
        direction = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(direction))
        if length < 1:
            return line

        angle = float(np.arctan2(direction[1], direction[0]))
        angle_difference = (angle - reference_angle + np.pi / 2) % np.pi - np.pi / 2
        max_change = np.deg2rad(max_change_degrees)
        angle_adjustment = np.clip(angle_difference, -max_change, max_change)
        midpoint = endpoints.mean(axis=0)
        while True:
            constrained_angle = reference_angle + angle_adjustment
            constrained_direction = np.array(
                [np.cos(constrained_angle), np.sin(constrained_angle)],
                dtype=np.float32,
            )
            half_vector = constrained_direction * (length / 2)
            constrained = np.array([midpoint - half_vector, midpoint + half_vector])
            result = tuple(tuple(np.round(point).astype(int)) for point in constrained)
            result_direction = np.asarray(result[1]) - result[0]
            result_angle = np.arctan2(result_direction[1], result_direction[0])
            result_difference = (
                (result_angle - reference_angle + np.pi / 2) % np.pi - np.pi / 2
            )
            if abs(result_difference) <= max_change or abs(angle_adjustment) < 1e-8:
                return result
            angle_adjustment -= np.sign(angle_adjustment) * min(
                np.deg2rad(0.1), abs(angle_adjustment)
            )

    @staticmethod
    def measure_line_polarity(gray, line, max_offset=3):
        endpoints = np.asarray(line, dtype=np.float32)
        direction = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(direction))
        if length < 1:
            return 0

        direction /= length
        normal = np.array([-direction[1], direction[0]])
        sample_count = min(100, max(12, int(length / 4)))
        sample_positions = np.linspace(0.1, 0.9, sample_count)
        samples = endpoints[0] + sample_positions[:, None] * (endpoints[1] - endpoints[0])
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        offsets = np.arange(-max_offset, max_offset + 1)
        locations = np.rint(
            samples[:, None, :] + normal[None, None, :] * offsets[None, :, None]
        ).astype(int)
        inside = (
            (locations[:, :, 0] >= 0)
            & (locations[:, :, 0] < gray.shape[1])
            & (locations[:, :, 1] >= 0)
            & (locations[:, :, 1] < gray.shape[0])
        )
        x = np.clip(locations[:, :, 0], 0, gray.shape[1] - 1)
        y = np.clip(locations[:, :, 1], 0, gray.shape[0] - 1)
        signed_strength = gradient_x[y, x] * normal[0] + gradient_y[y, x] * normal[1]
        signed_strength[~inside] = 0
        best_indices = np.argmax(np.abs(signed_strength), axis=1)
        best_strengths = signed_strength[np.arange(sample_count), best_indices]
        reliable = np.abs(best_strengths) >= 10.0
        if np.count_nonzero(reliable) < 3:
            return 0

        positive_fraction = float(np.mean(best_strengths[reliable] > 0))
        if positive_fraction >= 0.65:
            return 1
        if positive_fraction <= 0.35:
            return -1
        return 0

    @staticmethod
    def snap_line_to_edge(
        gray,
        line,
        max_offset=6,
        preserve_direction=False,
        preferred_polarity=0,
    ):
        endpoints = np.asarray(line, dtype=np.float32)
        direction = endpoints[1] - endpoints[0]
        length = np.linalg.norm(direction)
        if length < 1:
            return line

        direction /= length
        normal = np.array([-direction[1], direction[0]])
        sample_count = min(100, max(12, int(length / 4)))
        sample_positions = np.linspace(0.1, 0.9, sample_count)
        samples = endpoints[0] + sample_positions[:, None] * (endpoints[1] - endpoints[0])
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        offsets = np.arange(-max_offset, max_offset + 1)
        locations = np.rint(
            samples[:, None, :] + normal[None, None, :] * offsets[None, :, None]
        ).astype(int)
        inside = (
            (locations[:, :, 0] >= 0)
            & (locations[:, :, 0] < gray.shape[1])
            & (locations[:, :, 1] >= 0)
            & (locations[:, :, 1] < gray.shape[0])
        )
        x = np.clip(locations[:, :, 0], 0, gray.shape[1] - 1)
        y = np.clip(locations[:, :, 1], 0, gray.shape[0] - 1)
        signed_strength = gradient_x[y, x] * normal[0] + gradient_y[y, x] * normal[1]
        edge_strength = (
            np.maximum(signed_strength * preferred_polarity, 0)
            if preferred_polarity
            else np.abs(signed_strength)
        )
        edge_strength[~inside] = 0
        best_indices = np.argmax(edge_strength, axis=1)
        best_strengths = edge_strength[np.arange(sample_count), best_indices]
        threshold = max(10.0, float(np.median(best_strengths)) * 0.5)
        valid = best_strengths >= threshold
        if np.count_nonzero(valid) < 3:
            return line

        if preserve_direction:
            median_offset = float(np.median(offsets[best_indices[valid]]))
            snapped = endpoints + normal * median_offset
            return tuple(tuple(np.round(point).astype(int)) for point in snapped)

        edge_points = samples[valid] + normal * offsets[best_indices[valid], None]
        vx, vy, center_x, center_y = cv2.fitLine(
            edge_points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).flatten()
        edge_direction = np.array([vx, vy], dtype=np.float32)
        if np.dot(edge_direction, direction) < 0:
            edge_direction *= -1
        edge_center = np.array([center_x, center_y], dtype=np.float32)
        projections = (endpoints - edge_center) @ edge_direction
        snapped = edge_center + projections[:, None] * edge_direction
        return tuple(tuple(np.round(point).astype(int)) for point in snapped)

    @staticmethod
    def points_near_line(points, line, max_distance):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        start, end = np.asarray(line, dtype=np.float32)
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        if length_squared == 0:
            return np.zeros(len(points), dtype=bool)
        along = ((points - start) @ direction) / length_squared
        normal = np.array([-direction[1], direction[0]]) / np.sqrt(length_squared)
        distances = np.abs((points - start) @ normal)
        return (distances <= max_distance) & (along >= -0.15) & (along <= 1.15)

    @staticmethod
    def gap_segment(line1, line2):
        """Return the perpendicular connector between two approximately parallel lines."""
        first = np.asarray(line1, dtype=np.float32)
        second = np.asarray(line2, dtype=np.float32)
        direction1 = first[1] - first[0]
        direction2 = second[1] - second[0]
        direction1 /= max(np.linalg.norm(direction1), 1e-6)
        direction2 /= max(np.linalg.norm(direction2), 1e-6)
        direction = direction1 + direction2
        direction /= max(np.linalg.norm(direction), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        midpoint1 = first.mean(axis=0)
        midpoint2 = second.mean(axis=0)
        connector_end = midpoint1 + normal * np.dot(midpoint2 - midpoint1, normal)
        return tuple(np.round(midpoint1).astype(int)), tuple(np.round(connector_end).astype(int))

    @staticmethod
    def line_gap(line1, line2):
        connector_start, connector_end = CVTSheaveTracker.gap_segment(line1, line2)
        return float(np.linalg.norm(np.asarray(connector_end) - connector_start))

    @staticmethod
    def constrain_line_pair(line1, line2, min_gap=None, max_gap=None):
        first = np.asarray(line1, dtype=np.float32)
        second = np.asarray(line2, dtype=np.float32)
        direction1 = first[1] - first[0]
        direction2 = second[1] - second[0]
        direction1 /= max(np.linalg.norm(direction1), 1e-6)
        direction2 /= max(np.linalg.norm(direction2), 1e-6)
        direction = direction1 + direction2
        direction /= max(np.linalg.norm(direction), 1e-6)
        normal = np.array([-direction[1], direction[0]])
        signed_gap = float(np.dot(second.mean(axis=0) - first.mean(axis=0), normal))
        current_gap = abs(signed_gap)
        target_gap = current_gap
        side = 1 if signed_gap >= 0 else -1

        if min_gap is not None and current_gap < min_gap:
            target_gap = min_gap + 1
        if max_gap is not None and current_gap > max_gap:
            target_gap = max(0, max_gap - 1)

        if target_gap != current_gap:
            second += normal * side * (target_gap - current_gap)

        adjusted_second = tuple(
            tuple(np.round(point).astype(int)) for point in second
        )
        adjusted_first = tuple(
            tuple(np.round(point).astype(int)) for point in first
        )

        for _ in range(2):
            adjusted_gap = CVTSheaveTracker.line_gap(adjusted_first, adjusted_second)
            correction = 0.0
            if min_gap is not None and adjusted_gap < min_gap:
                correction = min_gap + 1 - adjusted_gap
            elif max_gap is not None and adjusted_gap > max_gap:
                correction = max(0, max_gap - 1) - adjusted_gap
            if correction == 0:
                break
            second = np.asarray(adjusted_second, dtype=np.float32)
            second += normal * side * correction
            adjusted_second = tuple(
                tuple(np.round(point).astype(int)) for point in second
            )

        return adjusted_first, adjusted_second

    def export_results(self, measurements):
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = results_dir / f"{Path(self.video_path).stem}_measurements_{timestamp}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(("frame_index", "timestamp_seconds", "primary_gap_in", "secondary_gap_in"))
            writer.writerows(measurements)

        print(f"Results exported to {output_path}")
        return output_path

    @staticmethod
    def run_plot_script():
        plot_script = Path(__file__).resolve().parent.parent / "results" / "plot_travel.py"
        try:
            result = subprocess.run([sys.executable, str(plot_script)], check=False)
        except OSError as error:
            print(f"Warning: could not run plot_travel.py: {error}")
            return
        if result.returncode != 0:
            print(f"Warning: plot_travel.py exited with code {result.returncode}.")

    def run(self):
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error reading video source.")
            return

        first_frame = self.resize_frame(first_frame)
        
        drawn_targets = [
            (f"Primary first sheave edge, away from belt (start {PRIMARY_MAX_GAP_IN:.3f} in)", 'primary_first', (0, 0, 255)),
            ("Primary second sheave edge, away from belt", 'primary_second', (0, 0, 255)),
            (f"Secondary first inner sheave edge, toward belt (start {SECONDARY_MIN_GAP_IN:.3f} in)", 'secondary_first', (255, 0, 0)),
            ("Secondary second inner sheave edge, toward belt", 'secondary_second', (255, 0, 0)),
            ("Belt edge 1 (draw along one side of a belt section)", 'belt_edge_first', (0, 255, 0)),
            ("Belt edge 2 (parallel, along the opposite side)", 'belt_edge_second', (0, 255, 0)),
        ]

        # Draw the sheave boundaries and belt edges used for local calibration.
        gray_first = self.apply_clahe(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
        if not self.select_tracking_lines(first_frame, gray_first, drawn_targets):
            self.cap.release()
            cv2.destroyAllWindows()
            return
        self.secondary_start_gap_px = self.line_gap(
            self.line_endpoints['secondary_first'],
            self.line_endpoints['secondary_second'],
        )
        if self.secondary_start_gap_px <= 0:
            print("Secondary lines overlap; draw them with a visible gap.")
            self.cap.release()
            cv2.destroyAllWindows()
            return
        self.primary_start_gap_px = self.line_gap(
            self.line_endpoints['primary_first'],
            self.line_endpoints['primary_second'],
        )
        if self.primary_start_gap_px <= 0:
            print("Primary lines overlap; draw them with a visible gap.")
            self.cap.release()
            cv2.destroyAllWindows()
            return

        self.belt_width_reference_px = self.line_gap(
            self.line_endpoints['belt_edge_first'],
            self.line_endpoints['belt_edge_second'],
        )
        if self.belt_width_reference_px <= 0:
            print("Draw two separated, parallel lines along opposite belt edges.")
            self.cap.release()
            cv2.destroyAllWindows()
            return

        self.belt_scale_factor = BELT_WIDTH_IN / self.belt_width_reference_px
        self.primary_local_scale_correction = PRIMARY_MAX_GAP_IN / (
            self.primary_start_gap_px * self.belt_scale_factor
        )
        self.secondary_local_scale_correction = SECONDARY_MIN_GAP_IN / (
            self.secondary_start_gap_px * self.belt_scale_factor
        )
        self.primary_scale_factor = (
            self.belt_scale_factor * self.primary_local_scale_correction
        )
        self.secondary_scale_factor = (
            self.belt_scale_factor * self.secondary_local_scale_correction
        )
        # Step 2: Main measurement loop
        win_name = "CVT Sheave Measurement"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        measurements = []
        frame_index = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.resize_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = self.apply_clahe(gray)
            selection_frame = frame
            portrait_layout = self.win_h > self.win_w

            if portrait_layout:
                overlay_height = 80
                status_box = (10, 8, self.win_w - 10, 36)
                status_position = (20, 28)
                measurement_box = (10, 44, self.win_w - 10, 72)
                measurement_position = (20, 64)
            else:
                overlay_height = 45
                status_box = (10, 8, 340, 37)
                status_position = (20, 28)
                measurement_box = (self.win_w - 360, 8, self.win_w - 10, 37)
                measurement_position = (self.win_w - 350, 28)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.win_w, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

            elapsed_fps = fps if fps > 0 else 30.0
            elapsed_seconds = int(frame_index / elapsed_fps)
            elapsed_hours, remainder = divmod(elapsed_seconds, 3600)
            elapsed_minutes, elapsed_seconds = divmod(remainder, 60)
            elapsed_text = (
                f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"
            )
            status_text = f"SPACE: Redraw lines | Elapsed: {elapsed_text}"
            status_font_scale = 0.50
            status_width = cv2.getTextSize(
                status_text, cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, 1
            )[0][0]
            status_available_width = status_box[2] - status_position[0] - 8
            if status_width > status_available_width:
                status_font_scale *= status_available_width / status_width
            cv2.rectangle(frame, status_box[:2], status_box[2:], (0, 0, 255), 2)
            cv2.putText(
                frame,
                status_text,
                status_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                status_font_scale,
                (255, 255, 255),
                1,
            )
            cv2.rectangle(frame, measurement_box[:2], measurement_box[2:], (255, 0, 0), 2)

            tracked_lines = {
                key: self.track_line(gray_eq, key)
                for key in self.line_endpoints
            }
            self.previous_gray = gray_eq

            tracked_lines['primary_first'], tracked_lines['primary_second'] = (
                self.constrain_line_pair(
                    tracked_lines['primary_first'],
                    tracked_lines['primary_second'],
                    min_gap=PRIMARY_MIN_GAP_IN / self.primary_scale_factor,
                    max_gap=PRIMARY_MAX_GAP_IN / self.primary_scale_factor,
                )
            )
            tracked_lines['secondary_first'], tracked_lines['secondary_second'] = (
                self.constrain_line_pair(
                    tracked_lines['secondary_first'],
                    tracked_lines['secondary_second'],
                    min_gap=SECONDARY_MIN_GAP_IN / self.secondary_scale_factor,
                )
            )
            for key in (
                'primary_first',
                'primary_second',
                'secondary_first',
                'secondary_second',
            ):
                self.line_endpoints[key] = tracked_lines[key]
            prim_gap_px = self.line_gap(
                tracked_lines['primary_first'], tracked_lines['primary_second']
            )
            sec_gap_px = self.line_gap(
                tracked_lines['secondary_first'], tracked_lines['secondary_second']
            )

            prim_gap_in = prim_gap_px * self.primary_scale_factor
            sec_gap_in = sec_gap_px * self.secondary_scale_factor
            frame_timestamp = frame_index / fps if fps > 0 else ""
            measurements.append((
                frame_index,
                frame_timestamp,
                round(prim_gap_in, 6),
                round(sec_gap_in, 6),
            ))
            frame_index += 1

            meas_text = f"Primary: {prim_gap_in:.3f} in | Sec: {sec_gap_in:.3f} in"
            measurement_font_scale = 0.50
            measurement_width = cv2.getTextSize(
                meas_text, cv2.FONT_HERSHEY_SIMPLEX, measurement_font_scale, 1
            )[0][0]
            measurement_available_width = (
                measurement_box[2] - measurement_position[0] - 8
            )
            if measurement_width > measurement_available_width:
                measurement_font_scale *= measurement_available_width / measurement_width
            cv2.putText(
                frame,
                meas_text,
                measurement_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                measurement_font_scale,
                (255, 255, 255),
                1,
            )

            # Show the currently tracked boundaries.
            for key, color in (
                ('primary_first', (0, 0, 255)),
                ('primary_second', (0, 0, 255)),
                ('secondary_first', (255, 0, 0)),
                ('secondary_second', (255, 0, 0)),
                ('belt_edge_first', (0, 255, 0)),
                ('belt_edge_second', (0, 255, 0)),
            ):
                start, end = tracked_lines[key]
                cv2.line(frame, start, end, color, 3)

            for first_key, second_key in (
                ('primary_first', 'primary_second'),
                ('secondary_first', 'secondary_second'),
                ('belt_edge_first', 'belt_edge_second'),
            ):
                gap_start, gap_end = self.gap_segment(
                    tracked_lines[first_key], tracked_lines[second_key]
                )
                cv2.line(frame, gap_start, gap_end, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == 32:
                cv2.destroyWindow(win_name)
                if not self.select_tracking_lines(
                    selection_frame,
                    gray_eq,
                    drawn_targets,
                    is_correction=True,
                ):
                    break
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            if not self.is_window_open(win_name):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.export_results(measurements)
        self.run_plot_script()

if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    tracker = CVTSheaveTracker(video_path)
    tracker.run()