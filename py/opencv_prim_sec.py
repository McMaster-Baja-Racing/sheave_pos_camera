import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BELT_WIDTH_IN = 0.905
PRIMARY_MIN_GAP_IN = BELT_WIDTH_IN
PRIMARY_MAX_GAP_IN = 1.7
SECONDARY_MIN_GAP_IN = BELT_WIDTH_IN


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

        screen_w = 1920
        screen_h = 1080
        self.win_w = min(1200, int(screen_w * 0.6))
        self.win_h = min(800, int(screen_h * 0.6))
        
        # Track line-selection and optical-flow state.
        self.drawing = False
        self.line_start = None
        self.current_line = None
        self.line_endpoints = {}
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

    def track_line(self, gray, key):
        """Track features but move the selected line only perpendicular to itself."""
        points = self.line_points[key]
        if self.previous_gray is not None and len(points) > 0:
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

        line = self.snap_line_to_edge(
            gray,
            self.line_endpoints[key],
            max_offset=6,
        )
        self.line_endpoints[key] = line
        self.initialize_line_points(gray, key)
        return self.line_endpoints[key]

    @staticmethod
    def snap_line_to_edge(gray, line, max_offset=6):
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
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

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
        edge_strength = np.abs(
            gradient_x[y, x] * normal[0] + gradient_y[y, x] * normal[1]
        )
        edge_strength[~inside] = 0
        best_indices = np.argmax(edge_strength, axis=1)
        best_strengths = edge_strength[np.arange(sample_count), best_indices]
        threshold = max(10.0, float(np.median(best_strengths)) * 0.5)
        valid = best_strengths >= threshold
        if np.count_nonzero(valid) < 3:
            return line

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

    def run(self):
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error reading video source.")
            return

        first_frame = cv2.resize(first_frame, (self.win_w, self.win_h))
        
        drawn_targets = [
            (f"Primary first boundary (start gap {PRIMARY_MAX_GAP_IN:.3f} in)", 'primary_first', (0, 0, 255)),
            ("Primary second boundary", 'primary_second', (0, 0, 255)),
            (f"Secondary first boundary (start gap {SECONDARY_MIN_GAP_IN:.3f} in)", 'secondary_first', (255, 0, 0)),
            ("Secondary second boundary", 'secondary_second', (255, 0, 0)),
            ("Belt edge 1 (draw along one side of a belt section)", 'belt_edge_first', (0, 255, 0)),
            ("Belt edge 2 (parallel, along the opposite side)", 'belt_edge_second', (0, 255, 0)),
        ]

        # Draw the sheave boundaries and belt edges used for local calibration.
        gray_first = self.apply_clahe(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
        for label, key, color in drawn_targets:
            line = self.draw_sheave_selection(label, first_frame, color)
            if line is None:
                self.cap.release()
                cv2.destroyAllWindows()
                return
            start, end = line
            if start[0] > end[0]:
                line = (end, start)
            self.line_endpoints[key] = line
            self.initialize_line_points(gray_first, key)

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
        self.previous_gray = gray_first

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

            frame = cv2.resize(frame, (self.win_w, self.win_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = self.apply_clahe(gray)

            # Overlay bar (Matches window mockup)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.win_w, 45), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

            # Red instruction box
            cv2.rectangle(frame, (10, 8), (340, 37), (0, 0, 255), 2)
            cv2.putText(frame, "STATUS: Tracking Selected Sheaves", (20, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            # Blue measurement box
            cv2.rectangle(frame, (self.win_w - 360, 8), (self.win_w - 10, 37), (255, 0, 0), 2)

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

            # Display real-time measurement readings in top-right blue box
            meas_text = f"Primary: {prim_gap_in:.3f} in | Sec: {sec_gap_in:.3f} in"
            cv2.putText(frame, meas_text, (self.win_w - 350, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

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
            if not self.is_window_open(win_name):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.export_results(measurements)

if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    tracker = CVTSheaveTracker(video_path)
    tracker.run()