import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from threading import Thread


def order_points(pts):
    # Sortiere die Punkte in konsistenter Reihenfolge: [tl, tr, br, bl]
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def fit_image_to_window(image, window_width, window_height, return_params=False):
    """
    Skaliert das Bild so, dass es vollständig in ein Fenster der Größe
    (window_width x window_height) passt, ohne das Seitenverhältnis zu ändern.
    Falls nötig, wird das Bild zentriert auf schwarzem Hintergrund (Letterbox).

    Bei return_params=True werden neben dem skalierten Bild zusätzlich
    (scale, x_offset, y_offset, new_w, new_h) zurückgegeben, um die Mauskoordinaten
    umrechnen zu können.
    """
    h, w = image.shape[:2]
    scale = min(window_width / w, window_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Schwarzes Canvas erzeugen
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Offsets berechnen, um das Bild zentriert einzufügen
    x_offset = (window_width - new_w) // 2
    y_offset = (window_height - new_h) // 2

    # Bild skalieren
    resized_image = cv2.resize(image, (new_w, new_h))

    # Resized-Bild ins Canvas einbetten
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    if return_params:
        return canvas, scale, x_offset, y_offset, new_w, new_h
    return canvas


class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analyzer by Ole Desens")
        self.video_path = None
        self.output_dir = None

        # ROI-Parameter
        self.roi_points = []
        self.M = None
        self.roi_width = None
        self.roi_height = None
        self.scale_x = None  # mm pro Pixel in X
        self.scale_y = None  # mm pro Pixel in Y

        self.label = tk.Label(root, text="Video Analyzer by Ole Desens", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_button.pack(pady=5)

        self.select_output_button = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_button.pack(pady=5)

        self.select_roi_button = tk.Button(root, text="Select ROI (4 Punkte)", command=self.select_roi)
        self.select_roi_button.pack(pady=5)

        self.start_button = tk.Button(root, text="Start Analysis", command=self.start_analysis, state=tk.DISABLED)
        self.start_button.pack(pady=5)

        self.ask_confirmation = tk.BooleanVar()
        self.ask_confirmation.set(False)
        self.ask_confirmation_checkbutton = tk.Checkbutton(root, text="Ask for confirmation",
                                                           variable=self.ask_confirmation)
        self.ask_confirmation_checkbutton.pack(pady=5)

        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress, maximum=100)
        self.progress_bar.pack(pady=5, fill=tk.X)

        self.particle_count_label = tk.Label(root, text="Particles detected: 0")
        self.particle_count_label.pack(pady=5)

        # DataFrame zum Speichern der Partikelinformationen
        self.particle_data = pd.DataFrame(columns=['ID', 'Frame', 'Area_pixels', 'Area_mm2', 'X', 'Y'])
        self.current_particle_id = 1

        # Konfigurations-Frame
        self.config_frame = tk.LabelFrame(root, text="Configuration", padx=10, pady=10)
        self.config_frame.pack(pady=5)

        self.threshold_label = tk.Label(self.config_frame, text="Threshold:")
        self.threshold_label.grid(row=0, column=0, sticky='e')
        self.threshold = tk.IntVar(value=50)
        self.threshold_entry = tk.Entry(self.config_frame, textvariable=self.threshold)
        self.threshold_entry.grid(row=0, column=1, pady=5)

        self.contour_area_label = tk.Label(self.config_frame, text="Min. Contour Area:")
        self.contour_area_label.grid(row=1, column=0, sticky='e')
        self.contour_area = tk.IntVar(value=20)
        self.contour_area_entry = tk.Entry(self.config_frame, textvariable=self.contour_area)
        self.contour_area_entry.grid(row=1, column=1, pady=5)

        self.auto_confirm_threshold_label = tk.Label(self.config_frame, text="Auto-Confirm Threshold:")
        self.auto_confirm_threshold_label.grid(row=2, column=0, sticky='e')
        self.auto_confirm_threshold = tk.IntVar(value=5)
        self.auto_confirm_threshold_entry = tk.Entry(self.config_frame, textvariable=self.auto_confirm_threshold)
        self.auto_confirm_threshold_entry.grid(row=2, column=1, pady=5)

        self.frame_skip_label = tk.Label(self.config_frame, text="Frame Skip:")
        self.frame_skip_label.grid(row=3, column=0, sticky='e')
        self.frame_skip = tk.IntVar(value=1)
        self.frame_skip_entry = tk.Entry(self.config_frame, textvariable=self.frame_skip)
        self.frame_skip_entry.grid(row=3, column=1, pady=5)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path and self.output_dir:
            self.start_button.config(state=tk.NORMAL)

    def select_output_folder(self):
        self.output_dir = filedialog.askdirectory()
        if self.video_path and self.output_dir:
            self.start_button.config(state=tk.NORMAL)

    def select_roi(self):
        if not self.video_path:
            messagebox.showwarning("Warnung", "Bitte zuerst ein Video auswählen.")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Konnte den ersten Frame nicht laden.")
            return

        clone = frame.copy()
        self.roi_points = []

        # Fest definierte Fenstergröße für die ROI-Auswahl
        roi_window_width = 1280
        roi_window_height = 50

        # Erzeuge das skalierte Vorschaubild und speichere Skalierung + Offsets
        display_image, scale, x_offset, y_offset, new_w, new_h = fit_image_to_window(
            clone, roi_window_width, roi_window_height, return_params=True
        )

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", int(roi_window_width), int(roi_window_height))
        cv2.imshow("Select ROI", display_image)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Prüfen, ob der Klick innerhalb des Bildbereichs liegt
                if x < x_offset or x > x_offset + new_w or y < y_offset or y > y_offset + new_h:
                    return
                # Mauskoordinaten zurück in Originalkoordinaten umrechnen
                orig_x = int((x - x_offset) / scale)
                orig_y = int((y - y_offset) / scale)

                self.roi_points.append([orig_x, orig_y])

                # Zur Kontrolle einen Kreis ins Original-Frame malen
                cv2.circle(clone, (orig_x, orig_y), 5, (0, 255, 0), -1)
                updated_display = fit_image_to_window(clone, roi_window_width, roi_window_height)
                cv2.imshow("Select ROI", updated_display)

                # Sobald 4 Punkte gesetzt wurden
                if len(self.roi_points) == 4:
                    pts = order_points(self.roi_points)
                    cv2.polylines(clone, [pts.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                    updated_display = fit_image_to_window(clone, roi_window_width, roi_window_height)
                    cv2.imshow("Select ROI", updated_display)
                    cv2.waitKey(500)
                    cv2.destroyWindow("Select ROI")

                    # ROI-Breite/Höhe berechnen
                    widthA = np.linalg.norm(pts[2] - pts[3])
                    widthB = np.linalg.norm(pts[1] - pts[0])
                    roi_width = int(max(widthA, widthB))

                    heightA = np.linalg.norm(pts[1] - pts[2])
                    heightB = np.linalg.norm(pts[0] - pts[3])
                    roi_height = int(max(heightA, heightB))

                    # Prüfung auf gültige Werte
                    if roi_width < 1 or roi_height < 1:
                        messagebox.showerror("ROI Error",
                                             "ROI-Dimension ist 0 oder zu klein. Bitte erneut Punkte wählen.")
                        return

                    self.roi_width = roi_width
                    self.roi_height = roi_height

                    dst = np.array([
                        [0, 0],
                        [roi_width - 1, 0],
                        [roi_width - 1, roi_height - 1],
                        [0, roi_height - 1]
                    ], dtype="float32")

                    self.M = cv2.getPerspectiveTransform(pts, dst)

                    # Beispielskalierung: ROI entspricht 120 mm in X, 3 mm in Y
                    self.scale_x = 120 / roi_width
                    self.scale_y = 3 / roi_height

                    messagebox.showinfo(
                        "ROI gesetzt",
                        f"ROI-Größe: {roi_width}px x {roi_height}px\n"
                        f"Skalierung: {self.scale_x:.3f} mm/px (X), {self.scale_y:.3f} mm/px (Y)"
                    )

        cv2.setMouseCallback("Select ROI", click_event)
        cv2.waitKey(0)

    def start_analysis(self):
        if self.M is None:
            messagebox.showwarning("Warnung", "Bitte zuerst den ROI (4 Punkte) setzen.")
            return
        self.progress.set(0)
        thread = Thread(target=self.analyze_video)
        thread.start()

    def ask_user_confirmation(self, image):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        scale_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        cv2.namedWindow("Detected Particle", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Particle", new_width, new_height)
        window_x = (screen_width - new_width) // 2
        window_y = (screen_height - new_height) // 2
        cv2.moveWindow("Detected Particle", window_x, window_y)
        cv2.imshow("Detected Particle", resized_image)
        cv2.waitKey(1)
        response = messagebox.askyesno("Particle Confirmation", "Is this particle correctly detected?")
        cv2.destroyWindow("Detected Particle")
        return response

    def save_intermediate_results(self, file_path):
        if not self.particle_data.empty:
            self.particle_data.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            self.particle_data = pd.DataFrame(columns=['ID', 'Frame', 'Area_pixels', 'Area_mm2', 'X', 'Y'])

    def analyze_video(self):
        try:
            video_cap = cv2.VideoCapture(self.video_path)
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_date = datetime.now().strftime("%Y%m%d")
            base_dir = os.path.join(self.output_dir, f"{current_date}_Objekterkennung")
            os.makedirs(base_dir, exist_ok=True)
            temp_csv_path = os.path.join(base_dir, f"{current_date}_intermediate_results.csv")

            # Erzeuge den MOG2-Background-Subtractor
            backSub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=30, detectShadows=False)

            # Fenster zur Anzeige: Feste Fenstergröße 1280x50 für die Videovorschau
            preview_width = 1280
            preview_height = 50
            cv2.namedWindow("Original ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Original ROI", int(preview_width), int(preview_height))
            cv2.namedWindow("Foreground Mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Foreground Mask", int(preview_width), int(preview_height))

            frame_number = 0
            total_particles = 0

            # Variable, die den letzten (verarbeiteten) Frame speichert
            previous_warped = None

            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                frame_number += 1

                # Aktuellen Frame transformieren
                current_warped = cv2.warpPerspective(
                    frame, self.M, (int(self.roi_width), int(self.roi_height))
                )

                # Wenn Frame-Skip aktiv ist, wird der Frame nicht ausgewertet,
                # aber previous_warped wird aktualisiert
                if frame_number % self.frame_skip.get() != 0:
                    previous_warped = current_warped
                    continue

                # MOG2-Background-Subtractor anwenden
                fgmask = backSub.apply(current_warped, learningRate=0.1)

                # Morphologische Operationen zur Rauschreduktion
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

                fgmask_colored = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

                # Konturen finden
                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > self.contour_area.get():
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / float(h)
                        if 0.5 < aspect_ratio < 2.0:
                            cv2.rectangle(fgmask_colored, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(fgmask_colored, f"ID: {self.current_particle_id}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # User-Confirmation
                            if (self.ask_confirmation.get() and
                                    self.current_particle_id <= self.auto_confirm_threshold.get()):
                                confirmed = self.ask_user_confirmation(fgmask_colored)
                            else:
                                confirmed = True

                            if confirmed:
                                # Ordner für diesen Partikel anlegen
                                particle_dir = os.path.join(base_dir, f"Particle_{self.current_particle_id}")
                                os.makedirs(particle_dir, exist_ok=True)

                                # Current Frame speichern
                                current_frame_filename = os.path.join(
                                    particle_dir, f"frame_{frame_number:04d}_current.png"
                                )
                                cv2.imwrite(current_frame_filename, current_warped)

                                # Previous Frame speichern (mit frame_number - 1)
                                if previous_warped is not None:
                                    prev_frame_filename = os.path.join(
                                        particle_dir, f"frame_{(frame_number - 1):04d}_previous.png"
                                    )
                                    cv2.imwrite(prev_frame_filename, previous_warped)

                                # Markiertes Bild
                                marked_filename = os.path.join(
                                    particle_dir, f"frame_{frame_number:04d}_marked.png"
                                )
                                cv2.imwrite(marked_filename, fgmask_colored)

                                # Maske mit gefüllter Kontur
                                mask_image_filename = os.path.join(
                                    particle_dir, f"frame_{frame_number:04d}_mask.png"
                                )
                                mask_image = current_warped.copy()
                                cv2.drawContours(mask_image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
                                cv2.imwrite(mask_image_filename, mask_image)

                                # Flächenberechnung
                                area_pixels = cv2.contourArea(contour)
                                area_mm2 = area_pixels * self.scale_x * self.scale_y

                                new_data = pd.DataFrame([{
                                    'ID': self.current_particle_id,
                                    'Frame': frame_number,
                                    'Area_pixels': area_pixels,
                                    'Area_mm2': area_mm2,
                                    'X': x,
                                    'Y': y
                                }])
                                self.particle_data = pd.concat([self.particle_data, new_data], ignore_index=True)
                                self.current_particle_id += 1
                                total_particles += 1

                                cv2.imshow("Foreground Mask",
                                           fit_image_to_window(fgmask_colored, preview_width, preview_height))
                                cv2.waitKey(100)

                # Vorschau aktualisieren
                preview_warped = fit_image_to_window(current_warped, preview_width, preview_height)
                preview_fgmask = fit_image_to_window(fgmask_colored, preview_width, preview_height)
                cv2.imshow("Original ROI", preview_warped)
                cv2.imshow("Foreground Mask", preview_fgmask)
                cv2.waitKey(1)

                # previous_warped auf den aktuellen Frame setzen (weil dieser verarbeitet wurde)
                previous_warped = current_warped

                if frame_number % 1000 == 0 and not self.particle_data.empty:
                    self.save_intermediate_results(temp_csv_path)
                self.progress.set((frame_number / total_frames) * 100)

            if not self.particle_data.empty:
                self.save_intermediate_results(temp_csv_path)

            if os.path.exists(temp_csv_path):
                final_particle_data = pd.read_csv(temp_csv_path)
                excel_filename = os.path.join(base_dir, f"{current_date}_particle_data.xlsx")
                final_particle_data.to_excel(excel_filename, index=False)
                os.remove(temp_csv_path)

            video_cap.release()
            cv2.destroyAllWindows()
            self.particle_count_label.config(text=f"Particles detected: {total_particles}")
            messagebox.showinfo("Analysis Complete", "The video analysis is complete.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.mainloop()
