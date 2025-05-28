import cv2
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox


def save_debug_image(debug_folder, filename, image):
    """Speichert ein Debug-Bild im angegebenen Ordner und gibt eine Rückmeldung."""
    if image is None or image.size == 0:
        print("Debug: Bild leer, nicht gespeichert:", filename)
        return
    full_path = os.path.join(debug_folder, filename)
    success = cv2.imwrite(full_path, image)
    if success:
        print("Debug: Bild gespeichert:", full_path)
    else:
        print("Debug: Fehler beim Speichern von:", full_path)


def extract_roi_from_marked(marked_img, margin=20):
    """
    Extrahiert den interessierenden Bereich (ROI) aus dem marked‑Bild.
    Der rote Bereich wird mittels HSV‑Segmentierung gesucht, die Bounding Box ermittelt
    und um den angegebenen Rand (margin) erweitert.
    """
    hsv = cv2.cvtColor(marked_img, cv2.COLOR_BGR2HSV)
    # Definiere HSV-Bereiche für Rot
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    # Schließe kleine Löcher in der Maske
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    nonzero = cv2.findNonZero(red_mask)
    if nonzero is None:
        return None
    x, y, w, h = cv2.boundingRect(nonzero)
    img_h, img_w = marked_img.shape[:2]
    new_x = max(x - margin, 0)
    new_y = max(y - margin, 0)
    new_w = min(w + 2 * margin, img_w - new_x)
    new_h = min(h + 2 * margin, img_h - new_y)
    return new_x, new_y, new_w, new_h, red_mask


def threshold_particle(roi):
    """
    Segmentiert das schwarze Partikel im ROI.
    Da in den Bildern das Partikel schwarz und der Hintergrund grau ist,
    wird THRESH_BINARY_INV verwendet – so erscheint das schwarze Partikel als weiße Fläche.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return thresh


def compute_difference_probability(thresh_prev, thresh_curr,
                                   noise_kernel_size=(3, 3),
                                   noise_iterations=2,
                                   area_threshold=20):
    """
    Erstellt ein Differenzbild aus den binären Threshold-Bildern von previous und current.
    Anschließend Rauschunterdrückung und Connected-Component-Auswertung:
    - Wenn eine zusammenhängende Fläche > area_threshold → probability = 1.0
    - Sonst: Fläche-zu-ROI-Relation (diff_area / roi_area, geclamped auf [0,1]).
    """
    # 1) Differenzbild
    diff = cv2.absdiff(thresh_prev, thresh_curr)
    # 2) Morphologisches Opening zur Rauschunterdrückung
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, noise_kernel_size)
    diff_clean = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=noise_iterations)

    # 3) Flächenberechnung
    diff_area = cv2.countNonZero(diff_clean)
    roi_area = thresh_prev.shape[0] * thresh_prev.shape[1]

    # 4) Connected Components auswerten
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_clean)
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # Index 0 = Hintergrund

    # 5) Entscheidung basierend auf CC-Größen
    if component_areas.size > 0 and np.any(component_areas > area_threshold):
        probability = 1.0
    else:
        probability = min(1.0, diff_area / roi_area) if roi_area > 0 else 0.0

    return diff, diff_clean, diff_area, probability


def process_particle_folder(folder_path, margin=20,
                            noise_kernel_size=(3, 3),
                            noise_iterations=2,
                            area_threshold=20):
    """
    Verarbeitet einen Particle-Unterordner:
      - Lädt Bilder previous, current, marked
      - Bestimmt ROI aus marked
      - Segmentiert Partikel, erstellt Diff-Bilder
      - Berechnet probability (0 oder 1) via CC-Logik
      - Speichert Debug-Bilder in Code_Ueberpruefung
    """
    # Dateipfade suchen
    previous_img_path = current_img_path = marked_img_path = None
    for file in os.listdir(folder_path):
        fname = file.lower()
        path = os.path.join(folder_path, file)
        if 'previous' in fname:
            previous_img_path = path
        elif 'current' in fname:
            current_img_path = path
        elif 'marked' in fname:
            marked_img_path = path

    if not (previous_img_path and current_img_path and marked_img_path):
        print(f"Notwendige Dateien in {folder_path} nicht gefunden.")
        return None, None, None

    prev = cv2.imread(previous_img_path)
    curr = cv2.imread(current_img_path)
    marked = cv2.imread(marked_img_path)
    if prev is None or curr is None or marked is None:
        print(f"Fehler beim Laden der Bilder in {folder_path}")
        return None, None, None

    # Debug-Ordner anlegen
    debug_folder = os.path.join(folder_path, 'Code_Ueberpruefung')
    os.makedirs(debug_folder, exist_ok=True)

    # ROI extrahieren
    roi_params = extract_roi_from_marked(marked, margin)
    if roi_params is None:
        print(f"Keine rote Markierung in {folder_path}")
        return None, None, None
    new_x, new_y, new_w, new_h, red_mask = roi_params
    save_debug_image(debug_folder, 'marked_red_mask.png', red_mask)
    marked_bbox = marked.copy()
    cv2.rectangle(marked_bbox, (new_x, new_y), (new_x+new_w, new_y+new_h), (0,255,0), 2)
    save_debug_image(debug_folder, 'marked_with_bbox.png', marked_bbox)

    # ROIs zuschneiden
    roi_prev = prev[new_y:new_y+new_h, new_x:new_x+new_w]
    roi_curr = curr[new_y:new_y+new_h, new_x:new_x+new_w]
    save_debug_image(debug_folder, 'roi_prev.png', roi_prev)
    save_debug_image(debug_folder, 'roi_curr.png', roi_curr)

    # Segmentierung
    thresh_prev = threshold_particle(roi_prev)
    thresh_curr = threshold_particle(roi_curr)
    save_debug_image(debug_folder, 'thresh_prev.png', thresh_prev)
    save_debug_image(debug_folder, 'thresh_curr.png', thresh_curr)

    # Differenz und Wahrscheinlichkeit
    diff, diff_clean, diff_area, probability = compute_difference_probability(
        thresh_prev, thresh_curr,
        noise_kernel_size=noise_kernel_size,
        noise_iterations=noise_iterations,
        area_threshold=area_threshold
    )
    save_debug_image(debug_folder, 'diff.png', diff)
    save_debug_image(debug_folder, 'diff_clean.png', diff_clean)
    print(f"Differenzfläche: {diff_area}, Detachment Probability: {probability}")

    return probability, diff_area, debug_folder


def main():
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Wähle den Ordner mit den Particle-Unterordnern")
    if not input_dir:
        messagebox.showwarning("Abbruch", "Kein Ordner ausgewählt.")
        return

    margin = simpledialog.askinteger("Parameter", "Erweiterungsrand (Pixel, Standard: 20):", initialvalue=20, parent=root) or 20
    area_threshold = simpledialog.askinteger("Parameter", "Mindestdiff-Fläche (Pixel, Standard: 20):", initialvalue=20, parent=root) or 20

    output_excel = filedialog.asksaveasfilename(
        title="Excel-Datei speichern unter", defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialfile="detachment_results.xlsx"
    )
    if not output_excel:
        messagebox.showwarning("Abbruch", "Kein Speicherort für Excel ausgewählt.")
        return

    particle_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                        if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("Particle_")]
    if not particle_folders:
        messagebox.showinfo("Info", "Keine Particle-Unterordner gefunden.")
        return

    results = []
    for folder in particle_folders:
        print("Verarbeite:", folder)
        probability, diff_area, _ = process_particle_folder(
            folder, margin,
            noise_kernel_size=(3,3), noise_iterations=2,
            area_threshold=area_threshold
        )
        if probability is not None:
            results.append({"Ordner": os.path.basename(folder),
                             "Detachment_Probability": probability,
                             "Differenz_Flaeche": diff_area})

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    messagebox.showinfo("Ergebnis", f"Ergebnisse wurden in {output_excel} gespeichert.")


if __name__ == '__main__':
    main()
