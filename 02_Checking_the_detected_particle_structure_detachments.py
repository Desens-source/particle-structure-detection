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
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return thresh


def compute_difference_probability(thresh_prev, thresh_curr, noise_kernel_size=(3, 3), noise_iterations=2):
    """
    Erstellt ein Differenzbild aus den binären Threshold-Bildern von previous und current.
    Es wird eine absolute Differenz berechnet, anschließend erfolgt eine Rauschunterdrückung
    mittels morphologischer Opening-Operation.

    Die "Differenzfläche" (diff_area) wird als die Anzahl der weißen Pixel im bereinigten
    Differenzbild ermittelt. Als detachment probability wird der Anteil der diff_area an der
    Gesamtfläche des ROIs verwendet (clamped auf [0,1]).
    """
    # Differenzbild berechnen (absolute Differenz der binären Bilder)
    diff = cv2.absdiff(thresh_prev, thresh_curr)
    # Rauschunterdrückung: Morphologisches Opening, um kleine Verschiebungen herauszufiltern
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, noise_kernel_size)
    diff_clean = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=noise_iterations)
    diff_area = cv2.countNonZero(diff_clean)
    # Gesamtfläche des ROIs
    roi_area = thresh_prev.shape[0] * thresh_prev.shape[1]
    if roi_area == 0:
        probability = 0.0
    else:
        probability = min(1.0, diff_area / roi_area)
    return diff, diff_clean, diff_area, probability


def process_particle_folder(folder_path, margin=20, noise_kernel_size=(3, 3), noise_iterations=2, area_threshold=20):
    """
    Verarbeitet einen Particle-Unterordner:
      - Lädt die Bilder "previous", "current" und "marked".
      - Nutzt das "marked"-Bild, um den interessierenden Bereich (ROI) zu bestimmen.
      - Schneidet den ROI aus den "previous" und "current" Bildern aus.
      - Segmentiert in den ROIs das schwarze Partikel mittels THRESH_BINARY_INV.
      - Erstellt ein Differenzbild aus den beiden binären Bildern.
      - Unterdrückt Rauschen mittels morphologischer Opening-Operation.
      - Berechnet die Differenzfläche und daraus die detachment probability.
      - Speichert alle Debug-Bilder im Unterordner "Code_Ueberpruefung".
    """
    # Dateisuche
    previous_img_path = None
    current_img_path = None
    marked_img_path = None
    for file in os.listdir(folder_path):
        fname = file.lower()
        if "previous" in fname:
            previous_img_path = os.path.join(folder_path, file)
        elif "current" in fname:
            current_img_path = os.path.join(folder_path, file)
        elif "marked" in fname:
            marked_img_path = os.path.join(folder_path, file)
    if previous_img_path is None or current_img_path is None or marked_img_path is None:
        print("Notwendige Dateien in", folder_path, "nicht gefunden.")
        return None, None, None

    prev = cv2.imread(previous_img_path)
    curr = cv2.imread(current_img_path)
    marked = cv2.imread(marked_img_path)
    if prev is None or curr is None or marked is None:
        print("Fehler beim Laden der Bilder in", folder_path)
        return None, None, None

    # Debug-Ordner anlegen (ohne Umlaute)
    debug_folder = os.path.join(folder_path, "Code_Ueberpruefung")
    os.makedirs(debug_folder, exist_ok=True)
    print("Debug-Ordner erstellt:", debug_folder)

    # ROI aus dem marked-Bild extrahieren
    roi_params = extract_roi_from_marked(marked, margin)
    if roi_params is None:
        print("Keine rote Markierung im marked-Bild gefunden in", folder_path)
        return None, None, None
    new_x, new_y, new_w, new_h, red_mask = roi_params
    save_debug_image(debug_folder, "marked_red_mask.png", red_mask)
    marked_bbox = marked.copy()
    cv2.rectangle(marked_bbox, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
    save_debug_image(debug_folder, "marked_with_bbox.png", marked_bbox)

    # ROI aus previous und current ausschneiden
    roi_prev = prev[new_y:new_y + new_h, new_x:new_x + new_w]
    roi_curr = curr[new_y:new_y + new_h, new_x:new_x + new_w]
    save_debug_image(debug_folder, "roi_prev.png", roi_prev)
    save_debug_image(debug_folder, "roi_curr.png", roi_curr)

    # Segmentierung mittels THRESH_BINARY_INV
    thresh_prev = threshold_particle(roi_prev)
    thresh_curr = threshold_particle(roi_curr)
    save_debug_image(debug_folder, "thresh_prev.png", thresh_prev)
    save_debug_image(debug_folder, "thresh_curr.png", thresh_curr)

    # Differenzbild aus den binären Bildern
    diff, diff_clean, diff_area, probability = compute_difference_probability(thresh_prev, thresh_curr,
                                                                              noise_kernel_size, noise_iterations)
    save_debug_image(debug_folder, "diff.png", diff)
    save_debug_image(debug_folder, "diff_clean.png", diff_clean)
    print("Differenzfläche:", diff_area, "ROI-Fläche:", thresh_prev.shape[0] * thresh_prev.shape[1])
    print("Detachment Probability (diff_area/ROI_area):", probability)

    # Optional: Nur als Detektion werten, wenn diff_area größer als ein Mindestwert ist
    if diff_area < area_threshold:
        probability = 0.0
        print("Diff area unter Schwellwert, keine Detektion.")

    return probability, diff_area, debug_folder


def main():
    # Tkinter-Fenster zur Auswahl des Hauptordners
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Wähle den Ordner mit den Particle-Unterordnern")
    if not input_dir:
        messagebox.showwarning("Abbruch", "Kein Ordner ausgewählt.")
        return

    # Optional: Parameter abfragen
    margin = simpledialog.askinteger("Parameter", "Erweiterungsrand (Pixel, Standard: 20):", initialvalue=20,
                                     parent=root)
    if margin is None:
        margin = 20
    # Parameter für Rauschunterdrückung
    noise_kernel_size = (3, 3)
    noise_iterations = 2
    # Mindestfläche für Diff (um kleine Verschiebungen auszuschließen)
    area_threshold = simpledialog.askinteger("Parameter", "Mindestdiff-Fläche (Pixel, Standard: 20):", initialvalue=20,
                                             parent=root)
    if area_threshold is None:
        area_threshold = 20

    output_excel = filedialog.asksaveasfilename(title="Excel-Datei speichern unter", defaultextension=".xlsx",
                                                filetypes=[("Excel files", "*.xlsx")],
                                                initialfile="detachment_results.xlsx")
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
        res = process_particle_folder(folder, margin, noise_kernel_size, noise_iterations, area_threshold)
        if res[0] is not None:
            probability, diff_area, debug_folder = res
            folder_name = os.path.basename(folder)
            results.append({
                "Ordner": folder_name,
                "Detachment_Probability": probability,
                "Differenz_Flaeche": diff_area
            })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    messagebox.showinfo("Ergebnis", f"Ergebnisse wurden in {output_excel} gespeichert.")


if __name__ == "__main__":
    main()
