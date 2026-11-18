import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure non-ASCII characters (e.g. a Hebrew folder name) print correctly in the Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent


def find_data_dir():
    """Find the task folder (the folder that contains train.csv, test.csv and images/).

    Looked up in this order:
      1. A folder given on the command line:  python analysis.py "C:\\path\\to\\task_folder"
      2. The folder of this script
      3. Any sub-folder of this script's folder (e.g. the extracted task folder)
      4. The task folder on the author's Desktop
    """
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]))
    candidates.append(SCRIPT_DIR)
    candidates.extend(sorted(p for p in SCRIPT_DIR.iterdir() if p.is_dir()))
    candidates.append(Path.home() / "OneDrive" / "Desktop" / "משימה")

    for folder in candidates:
        if (folder / "train.csv").exists() and (folder / "test.csv").exists() and (folder / "images").is_dir():
            return folder
    sys.exit(
        "Task folder not found (a folder with train.csv, test.csv and images/).\n"
        "Put analysis.py next to the task folder, or run:  python analysis.py \"<path to task folder>\""
    )


DATA_DIR = find_data_dir()
DATA_PATH = DATA_DIR / "train.csv"
print(f"Using task folder: {DATA_DIR}")

# Show all rows and columns when printing
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# Pop-up windows: every message / chart opens in its own window,
# and the program continues only after the window is closed.
# ---------------------------------------------------------------
CLOSE_NOTE = "Close this window to continue"
MAX_WINDOW_HEIGHT_IN = 7  # keep tall charts inside the screen
MESSAGE_WIDTH_IN = 11.5  # wide text windows -> fewer lines -> shorter windows that fit the screen
MESSAGE_FONT_SIZE = 11
LINE_SPACING = 1.5


def show_message_and_wait(title, message):
    """Open a window with a text message; the program waits until it is closed."""
    wrapped = "\n\n".join(textwrap.fill(paragraph, width=125) for paragraph in message.split("\n"))

    # Window height follows the real height of the text (in inches), with a clear gap
    # between the text and the "close" note at the bottom.
    title_in, gap_in, note_in = 0.9, 0.4, 0.7
    fig_msg = plt.figure(figsize=(MESSAGE_WIDTH_IN, 6))
    body = fig_msg.text(0.05, 0.5, wrapped, fontsize=MESSAGE_FONT_SIZE, va="top", linespacing=LINE_SPACING)
    text_height_in = body.get_window_extent(fig_msg.canvas.get_renderer()).height / fig_msg.dpi

    fig_height_in = title_in + text_height_in + gap_in + note_in
    fig_msg.set_size_inches(MESSAGE_WIDTH_IN, fig_height_in)
    body.set_position((0.05, 1 - title_in / fig_height_in))
    fig_msg.text(0.05, 1 - 0.35 / fig_height_in, title, fontsize=15, fontweight="bold", va="top")
    fig_msg.text(0.5, 0.3 / fig_height_in, CLOSE_NOTE, ha="center", fontsize=12,
                 fontweight="bold", color="#555555")
    fig_msg.canvas.manager.set_window_title(title)
    plt.show()  # blocks until the window is closed
    plt.close(fig_msg)


def show_and_wait(image_path, window_title):
    """Open a saved chart in its own window with a 'close to continue' note at the bottom.

    The chart is shown from the saved PNG, so the note does not end up in the saved file.
    The program waits here until the window is closed.
    """
    image = plt.imread(image_path)
    height_px, width_px = image.shape[:2]
    note_height_in = 0.6
    image_height_in = min(10 * height_px / width_px, MAX_WINDOW_HEIGHT_IN)
    image_width_in = image_height_in * width_px / height_px
    fig_height_in = image_height_in + note_height_in

    fig_show = plt.figure(figsize=(image_width_in, fig_height_in))
    ax_show = fig_show.add_axes((0, note_height_in / fig_height_in, 1, image_height_in / fig_height_in))
    ax_show.imshow(image)
    ax_show.axis("off")
    fig_show.text(0.5, note_height_in / 2 / fig_height_in, CLOSE_NOTE,
                  ha="center", va="center", fontsize=12, fontweight="bold", color="#555555")
    fig_show.canvas.manager.set_window_title(window_title)
    plt.show()  # blocks until the window is closed
    plt.close(fig_show)


INTRO_MESSAGE = (
    "I started by looking at the raw data and tried to find consistency or a pattern. "
    "Very quickly I noticed that the time and the day/night shifts do not match, "
    "so I changed them (night starts at 18:00). Then I printed the data again and looked at it.\n"
    "I divided the data by conditions that make sense in the real world: batches (maybe one batch "
    "is defective), day and night (tired workers make more mistakes), and camera (maybe the camera "
    "has dirt on it or is broken).\n"
    "Then I moved to the more advanced analysis: I separated the data by batch and by day and night, "
    "and found the percentage of non-stain defects (stain defects are treated as \"noise\" "
    "and not a real problem).\n"
    "I found that Batch C has the highest percentage of non-stain defects, both day and night, "
    "which led me to believe that this batch is damaged."
)

print("How I approached the data:")
print(INTRO_MESSAGE)
show_message_and_wait("How I approached the data", INTRO_MESSAGE)

df = pd.read_csv(DATA_PATH)

print(df)

# ---------------------------------------------------------------
# 0. Fix the shift column: derive DAY / NIGHT from the actual timestamp
#    The original "shift" column does not match the capture time
#    (e.g. NIGHT at 07:00, DAY at 21:00), so we recompute it.
#    Rule: from 18:00 onwards = NIGHT, otherwise DAY (timestamps are UTC).
# ---------------------------------------------------------------
NIGHT_START_HOUR = 18

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df["shift_original"] = df["shift"]
df["shift"] = df["timestamp_utc"].dt.hour.map(
    lambda hour: "NIGHT" if hour >= NIGHT_START_HOUR else "DAY"
)

mismatches = (df["shift"] != df["shift_original"]).sum()

print("\n" + "=" * 50)
print("The day and night stamp does not match the actual details")
print("=" * 50)
print(f"Shift recomputed from timestamp_utc (NIGHT = from {NIGHT_START_HOUR}:00).")
print(f"Rows where the original shift was wrong: {mismatches} of {len(df)}")
print("\nOriginal shift vs. corrected shift:")
print(pd.crosstab(df["shift_original"], df["shift"], margins=True, margins_name="Total"))
print("\nCorrected table:")
print(df)

# All analyses below use the corrected "shift" column

LABEL_NAMES = {0: "OK", 1: "Defective"}

# ---------------------------------------------------------------
# 1. Number of records and label distribution (OK / Defective)
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print(f"Total images: {len(df)}")

label_counts = df["inspector_label"].map(LABEL_NAMES).value_counts()
label_dist = pd.DataFrame({
    "count": label_counts,
    "percent": (label_counts / len(df) * 100).round(1),
})
print("\nLabel distribution:")
print(label_dist)

# ---------------------------------------------------------------
# 2. Breakdown by camera: OK, Defective, total and defect rate
# ---------------------------------------------------------------
by_camera = pd.crosstab(df["camera"], df["inspector_label"]).rename(columns=LABEL_NAMES)
by_camera["Total"] = by_camera.sum(axis=1)
by_camera["Defect %"] = (by_camera["Defective"] / by_camera["Total"] * 100).round(1)

print("\n" + "=" * 50)
print("Breakdown by camera:")
print(by_camera)

# ---------------------------------------------------------------
# 3. Breakdown by camera and shift (DAY / NIGHT)
# ---------------------------------------------------------------
by_camera_shift = (
    pd.crosstab([df["camera"], df["shift"]], df["inspector_label"])
    .rename(columns=LABEL_NAMES)
)
by_camera_shift["Total"] = by_camera_shift.sum(axis=1)
by_camera_shift["Defect %"] = (
    by_camera_shift["Defective"] / by_camera_shift["Total"] * 100
).round(1)

print("\n" + "=" * 50)
print("Breakdown by camera and shift:")
for camera, table in by_camera_shift.groupby(level="camera"):
    print(f"\n{camera}:")
    print(table.droplevel("camera"))

# ---------------------------------------------------------------
# 4. Breakdown by batch: count of each inspector_type
# ---------------------------------------------------------------
by_batch = pd.crosstab(df["batch"], df["inspector_type"])
defect_types = [col for col in by_batch.columns if col != "none"]
by_batch = by_batch[["none"] + defect_types]  # put "none" (OK) first
by_batch["OK"] = by_batch["none"]
by_batch["Defective"] = by_batch[defect_types].sum(axis=1)
by_batch["Total"] = by_batch["OK"] + by_batch["Defective"]
by_batch["Defect %"] = (by_batch["Defective"] / by_batch["Total"] * 100).round(1)

print("\n" + "=" * 50)
print("Breakdown by batch (inspector_type counts):")
print(by_batch)

# ---------------------------------------------------------------
# 5. Breakdown by batch and shift (DAY / NIGHT)
# ---------------------------------------------------------------
by_batch_shift = (
    pd.crosstab([df["batch"], df["shift"]], df["inspector_label"])
    .rename(columns=LABEL_NAMES)
)
by_batch_shift["Total"] = by_batch_shift.sum(axis=1)
by_batch_shift["Defect %"] = (
    by_batch_shift["Defective"] / by_batch_shift["Total"] * 100
).round(1)

print("\n" + "=" * 50)
print("Breakdown by batch and shift:")
for batch, table in by_batch_shift.groupby(level="batch"):
    print(f"\n{batch}:")
    print(table.droplevel("batch"))

# ---------------------------------------------------------------
# 6. Stains per batch and shift: which camera / exposure setting?
#    Checks whether stains come from the same camera or exposure setting.
# ---------------------------------------------------------------
stains = df[df["inspector_type"] == "stain"]

print("\n" + "=" * 50)
print(f"Stains by batch and shift (total stains: {len(stains)})")
for batch, group in stains.groupby("batch"):
    print(f"\n{batch} - {len(group)} stains")

    print("\n  By camera:")
    print(pd.crosstab(group["shift"], group["camera"], margins=True, margins_name="Total"))

    print("\n  By exposure setting:")
    print(pd.crosstab(group["shift"], group["exposure_setting"], margins=True, margins_name="Total"))

# ---------------------------------------------------------------
# 7. Non-stain defects (edge_chip, missing_hole, scratch) by batch and shift
#    Stained images are removed completely: from the defects AND from the
#    total number of components, so the rate is
#    non-stain defects / (all images in the group - stained images).
# ---------------------------------------------------------------
df_no_stains = df[df["inspector_type"] != "stain"]
non_stain = df_no_stains[df_no_stains["inspector_type"] != "none"]

by_batch_shift_ns = pd.crosstab([non_stain["batch"], non_stain["shift"]], non_stain["inspector_type"])
by_batch_shift_ns["Non-stain defects"] = by_batch_shift_ns.sum(axis=1)

# Images in each batch/shift group excluding stained images, to turn counts into rates
group_sizes = df_no_stains.groupby(["batch", "shift"]).size()
by_batch_shift_ns = by_batch_shift_ns.reindex(group_sizes.index, fill_value=0)
by_batch_shift_ns["Images (excl. stains)"] = group_sizes
by_batch_shift_ns["Non-stain %"] = (
    by_batch_shift_ns["Non-stain defects"] / by_batch_shift_ns["Images (excl. stains)"] * 100
).round(1)

print("\n" + "=" * 50)
print(f"Non-stain defects by batch and shift (stained images removed: {len(df) - len(df_no_stains)})")
for batch, table in by_batch_shift_ns.groupby(level="batch"):
    print(f"\n{batch}:")
    print(table.droplevel("batch"))

# ---------------------------------------------------------------
# 8. Chart: non-stain defect % by batch, DAY vs NIGHT side by side
#    (stained images removed from both the defects and the totals)
# ---------------------------------------------------------------
SHIFT_COLORS = {"DAY": "#1f77b4", "NIGHT": "#ff7f0e"}  # blue, orange
BAR_WIDTH = 0.38
BAR_GAP = 0.02  # small gap between the DAY and NIGHT bars

CONCLUSIONS = (
    "I presented a graph of the non-stain defect percentage by batch and shift. "
    "I removed the stains because they can depend heavily on the camera, and a stain "
    "is a negligible defect anyway. We can see that Batch C's percentage of non-stain "
    "defects is high, with no relation to the time (day or night), and we can infer "
    "that this batch is damaged."
)

defect_pct = by_batch_shift_ns["Non-stain %"].unstack("shift")  # rows: batch, columns: DAY / NIGHT
totals = by_batch_shift_ns["Images (excl. stains)"].unstack("shift")
overall_pct = len(non_stain) / len(df_no_stains) * 100
batches = defect_pct.index.tolist()
x = range(len(batches))

fig, ax = plt.subplots(figsize=(9, 6.5))

for i, shift in enumerate(["DAY", "NIGHT"]):
    offset = (i - 0.5) * (BAR_WIDTH + BAR_GAP)
    positions = [pos + offset for pos in x]
    bars = ax.bar(positions, defect_pct[shift], width=BAR_WIDTH,
                  color=SHIFT_COLORS[shift], label=shift.capitalize(), zorder=2)
    # Label each bar with its % and number of images (groups are small)
    for bar, pct, n in zip(bars, defect_pct[shift], totals[shift]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9, color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=3)

ax.axhline(overall_pct, color="#888888", linestyle="--", linewidth=1, zorder=1,
           label=f"Overall ({overall_pct:.1f}%)")

ax.set_xticks(list(x))
ax.set_xticklabels(batches)
ax.set_xlabel("Batch")
ax.set_ylabel("Non-stain defect %")
ax.set_ylim(0, 60)
ax.set_title("Non-stain defects by batch and shift")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color="#e5e5e5", zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Conclusions below the chart, medium font size
fig.text(0.06, 0.02, CONCLUSIONS, ha="left", va="bottom", fontsize="medium", wrap=True)
fig.tight_layout(rect=(0, 0.17, 1, 1))  # leave room at the bottom for the conclusions

CHART_PATH = Path(__file__).parent / "non_stain_defects_by_batch_shift.png"
fig.savefig(CHART_PATH, dpi=150)
plt.close(fig)
print(f"\nChart saved to: {CHART_PATH}")

print("\nConclusions:")
print(CONCLUSIONS)
show_and_wait(CHART_PATH, "Non-stain defects by batch and shift")

print("\n" + "#" * 50)
print("END OF PART 1, START OF PART 2")
print("#" * 50)

# ===============================================================
# PART 2 - Understanding the images
# ===============================================================

# ---------------------------------------------------------------
# 1. Show 6 sample images: 3 OK and 3 Defective (each a different defect type)
# ---------------------------------------------------------------
IMAGES_DIR = DATA_PATH.parent  # the "file" column is relative to the task folder
RANDOM_SEED = 42  # fixed seed so the same samples are shown on every run

ok_samples = df[df["inspector_label"] == 0].sample(3, random_state=RANDOM_SEED)
defect_samples = (
    df[df["inspector_label"] == 1]
    .groupby("inspector_type")
    .sample(1, random_state=RANDOM_SEED)  # one image per defect type
    .sample(3, random_state=RANDOM_SEED)  # then pick 3 of the types
)
samples = pd.concat([ok_samples, defect_samples])

print("\nSample images:")
print(samples[["image_id", "batch", "camera", "shift", "exposure_setting",
               "inspector_label", "inspector_type"]].to_string(index=False))

fig_samples, axes = plt.subplots(2, 3, figsize=(10, 8), layout="constrained")
for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    image = plt.imread(IMAGES_DIR / row["file"])
    ax.imshow(image)
    label = LABEL_NAMES[row["inspector_label"]]
    title = f"{row['image_id']} - {label}"
    if row["inspector_label"] == 1:
        title += f" ({row['inspector_type']})"
    title += f"\n{row['camera']}, exposure {row['exposure_setting']}"
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig_samples.suptitle("Sample images: top row OK, bottom row Defective")
fig_samples.supxlabel(
    "One of the components is missing a hole, and the other one has an edge chip.",
    fontsize="medium",
)

SAMPLES_PATH = Path(__file__).parent / "sample_images.png"
fig_samples.savefig(SAMPLES_PATH, dpi=150)
plt.close(fig_samples)
print(f"\nSample images saved to: {SAMPLES_PATH}")

SAMPLES_MESSAGE = "Those are the 6 example pictures."
print(SAMPLES_MESSAGE)
show_message_and_wait("Sample images", SAMPLES_MESSAGE)

show_and_wait(SAMPLES_PATH, "Sample images")

# ---------------------------------------------------------------
# 2. Feature function: mean brightness and number of edge pixels
# ---------------------------------------------------------------
CANNY_LOW, CANNY_HIGH = 50, 150  # Canny edge detector thresholds


def extract_features(image_path):
    """Read an image and return (mean brightness, number of edge pixels).

    - Mean brightness: average gray level of the whole image (0 = black, 255 = white).
    - Edge count: number of pixels marked as edges by the Canny edge detector.
      Defects such as scratches, stains and chips add new lines/borders to the image.
    """
    # cv2.imread cannot open paths with non-ASCII characters on Windows, so decode from bytes
    gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    mean_brightness = float(gray.mean())

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise so it is not counted as edges
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edge_count = int(np.count_nonzero(edges))

    return mean_brightness, edge_count


# ---------------------------------------------------------------
# 3. Compute the features for all train images and compare OK vs Defective
# ---------------------------------------------------------------
features = [extract_features(IMAGES_DIR / file) for file in df["file"]]
df["mean_brightness"] = [f[0] for f in features]
df["edge_count"] = [f[1] for f in features]
df["label_name"] = df["inspector_label"].map(LABEL_NAMES)

print("\n" + "=" * 50)
print("Image features by label (OK vs Defective):")
print(
    df.groupby("label_name")[["mean_brightness", "edge_count"]]
    .agg(["count", "mean", "median", "std"])
    .round(1)
    .T
)

# Comparison chart: mean brightness, OK vs Defective
groups = ["OK", "Defective"]
group_colors = {"OK": "#1f77b4", "Defective": "#ff7f0e"}
values = [df.loc[df["label_name"] == g, "mean_brightness"] for g in groups]

BRIGHTNESS_NOTE = (
    "According to the data and statistics, we cannot separate the good components\n"
    "from the defective ones by brightness alone."
)

fig_feat, ax = plt.subplots(figsize=(7, 6))

# Show every image as a point, spread sideways so points do not hide each other
rng = np.random.default_rng(RANDOM_SEED)
for i, (group, vals) in enumerate(zip(groups, values), start=1):
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(i + jitter, vals, s=14, alpha=0.6, color=group_colors[group], zorder=3)
    ax.text(i + 0.3, vals.median(), f"median {vals.median():.1f}\n(n={len(vals)})",
            va="center", fontsize=9, color="#333333")

ax.set_xticks([1, 2])
ax.set_xticklabels(groups)
ax.set_xlim(0.5, 2.8)
ax.set_ylabel("Mean brightness (0 = black, 255 = white)")
ax.set_title("Mean brightness: OK vs Defective (train)")
ax.grid(axis="y", color="#e5e5e5", zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Note below the chart
fig_feat.text(0.5, 0.02, BRIGHTNESS_NOTE, ha="center", va="bottom", fontsize="medium")
fig_feat.tight_layout(rect=(0, 0.1, 1, 1))  # leave room at the bottom for the note

FEATURE_CHART_PATH = Path(__file__).parent / "brightness_ok_vs_defective.png"
fig_feat.savefig(FEATURE_CHART_PATH, dpi=150)
plt.close(fig_feat)
print(f"\nFeature comparison chart saved to: {FEATURE_CHART_PATH}")

FEATURES_MESSAGE = (
    "The 2 parameters I chose to check between the good pictures and the defective pictures "
    "are brightness and number of edges. I did a comparison and chose to present the brightness "
    "scale of the good and the defective pictures. I show it in the next graph."
)
print("\nAbout the features:")
print(FEATURES_MESSAGE)
show_message_and_wait("Image features: brightness and number of edges", FEATURES_MESSAGE)

show_and_wait(FEATURE_CHART_PATH, "Mean brightness: OK vs Defective")

print("\n" + "#" * 50)
print("END OF PART 2, START OF PART 3")
print("#" * 50)

# ===============================================================
# PART 3 - Decision rule for suspected defects
# ===============================================================

RULE_MESSAGE = (
    "I chose a combination of 3 parameters to check whether the picture is good or defective:\n"
    "1. Number of edges - 0 if the number is correct (4), 1 if defective (edge chip).\n"
    "2. Number of holes - 0 if the number is correct (9), 1 if not (missing hole).\n"
    "3. Sharp difference from the median - between 0 and 1. Because the brightness increases "
    "or decreases the brightness of every pixel by the same amount, the distance from the median "
    "will be more accurate (to find scratches).\n"
    "We take the maximum, so the clear defects will be flagged easily, and the scratches and stains "
    "will be more detectable. If the score is 0.5 or above, we flag it as a defect."
)
print("\nAbout the decision rule:")
print(RULE_MESSAGE)
show_message_and_wait("The decision rule", RULE_MESSAGE)

# ---------------------------------------------------------------
# Step A: shape features for every image (train and test)
#   - holes:         number of holes found (a normal component has 9)
#   - outline_edges: corners of the component outline (a normal square has 4;
#                    an edge chip cuts a notch and adds corners)
#   - dark_fraction: share of the component's face covered by dark pixels
#                    (holes, screws, lines and text cover an almost constant share;
#                    a stain adds dark area, a missing hole removes some)
# The thresholds are recomputed for every image, so bright (flash) or dark images
# do not change the result.
# ---------------------------------------------------------------
TEST_PATH = DATA_PATH.parent / "test.csv"

HOLE_MIN_AREA, HOLE_MAX_AREA = 150, 500  # pixel area of one hole (~300 px)
HOLE_MIN_CIRCULARITY = 0.7  # 1.0 = perfect circle
HOLE_MAX_CENTER_DIST = 0.45  # holes are near the center; screws are in the corners (~0.58)
POLY_EPSILON_RATIO = 0.01  # outline simplification tolerance, as a fraction of the perimeter
BORDER_MARGIN_PX = 7  # ignore a thin band at the face border when measuring dark area


def extract_shape_features(image_path):
    """Read an image and return (holes, outline_edges, dark_fraction)."""
    # cv2.imread cannot open paths with non-ASCII characters on Windows, so decode from bytes
    gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Two-level threshold: background < grey rim < light face of the component.
    # Otsu picks each threshold automatically, so it adapts to dark and bright images.
    t_plate, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_pixels = gray[gray > t_plate].reshape(-1, 1)
    t_face, _ = cv2.threshold(plate_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    face = ((gray > t_face) * 255).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Outline edges: simplify the outer outline of the face to a polygon and count its corners
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(outline, POLY_EPSILON_RATIO * cv2.arcLength(outline, True), True)
    outline_edges = len(polygon)

    # Holes: dark blobs inside the face that are round and close to the center
    filled_face = np.zeros_like(face)  # everything inside the outline
    cv2.drawContours(filled_face, [outline], -1, 255, cv2.FILLED)
    dark_inside = cv2.bitwise_and(filled_face, cv2.bitwise_not(face))
    blobs, _ = cv2.findContours(dark_inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.moments(outline)
    center_x, center_y = m["m10"] / m["m00"], m["m01"] / m["m00"]
    face_size = np.sqrt(cv2.contourArea(outline))
    holes = 0
    for blob in blobs:
        area = cv2.contourArea(blob)
        if not HOLE_MIN_AREA <= area <= HOLE_MAX_AREA:
            continue
        circularity = 4 * np.pi * area / cv2.arcLength(blob, True) ** 2
        bm = cv2.moments(blob)
        dist = np.hypot(bm["m10"] / bm["m00"] - center_x, bm["m01"] / bm["m00"] - center_y) / face_size
        if circularity >= HOLE_MIN_CIRCULARITY and dist <= HOLE_MAX_CENTER_DIST:
            holes += 1

    # Dark fraction: share of dark pixels inside the face, ignoring a thin border band
    kernel = np.ones((BORDER_MARGIN_PX, BORDER_MARGIN_PX), np.uint8)
    inner = cv2.erode(filled_face, kernel) > 0
    dark_fraction = float(((gray < t_face) & inner).sum() / inner.sum())

    return holes, outline_edges, dark_fraction


def add_shape_features(table):
    """Return a copy of the table with holes, outline_edges and dark_fraction columns."""
    result = table.copy()
    values = [extract_shape_features(IMAGES_DIR / file) for file in result["file"]]
    result["holes"] = [v[0] for v in values]
    result["outline_edges"] = [v[1] for v in values]
    result["dark_fraction"] = [v[2] for v in values]
    return result


train = add_shape_features(df)
test = add_shape_features(pd.read_csv(TEST_PATH))

print("\n" + "=" * 50)
print(f"Step A - shape features computed: {len(train)} train images, {len(test)} test images")

print("\nTrain - holes found, by inspector_type:")
print(pd.crosstab(train["inspector_type"], train["holes"]))

print("\nTrain - outline edges found, by inspector_type:")
print(pd.crosstab(train["inspector_type"], train["outline_edges"]))

print("\nTrain - dark fraction (%), by inspector_type:")
print((train.groupby("inspector_type")["dark_fraction"]
       .describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]] * [1, 100, 100, 100])
      .round(2))

print("\nTest - feature summary (no labels):")
print(f"  holes:         {test['holes'].value_counts().sort_index().to_dict()}")
print(f"  outline_edges: {test['outline_edges'].value_counts().sort_index().to_dict()}")
print(f"  dark_fraction: median {test['dark_fraction'].median() * 100:.2f}%")

# ---------------------------------------------------------------
# Step B: split the labeled data, define the decision rule, check it on the development set
#   - 70% development / 30% validation, stratified (same OK/Defective ratio in both)
#   - The validation set is NOT used here; it is kept for evaluation (Part 4)
#   - test.csv is never used to evaluate or tune the rule
# ---------------------------------------------------------------
DEV_FRACTION = 0.7
NORMAL_HOLES = 9  # a normal component has 9 holes
NORMAL_OUTLINE_EDGES = 4  # a normal component is a square: 4 edges
DECISION_THRESHOLD = 0.5  # score >= threshold -> suspected defect

# Dark-area grading (only grades the score of images without a shape defect)
DARK_Z_LOW = 2  # up to 2 "normal spreads" from the typical value -> dark_score 0
DARK_Z_HIGH = 6  # 6 spreads or more -> maximum dark_score
DARK_SCORE_MAX = 0.49  # kept below the decision threshold, so it never decides alone

dev = train.groupby("inspector_label").sample(frac=DEV_FRACTION, random_state=RANDOM_SEED)
val = train.drop(dev.index)

# Typical dark fraction and its normal spread - learned from OK images of the development set only.
# The spread uses the median absolute deviation (MAD), which is not pulled by outliers;
# multiplying by 1.4826 puts it on the same scale as a standard deviation.
dev_ok_dark = dev.loc[dev["inspector_label"] == 0, "dark_fraction"]
DARK_TYPICAL = dev_ok_dark.median()
DARK_SPREAD = 1.4826 * (dev_ok_dark - DARK_TYPICAL).abs().median()


def defect_score(table):
    """Suspicion score between 0 and 1.

    - Shape defect (fewer than 9 holes, or more than 4 outline edges): score = 1.
    - Otherwise the score is graded by how unusual the dark area is:
        distance = |dark_fraction - typical| / spread
        distance <= 2 -> 0, distance >= 6 -> 0.49, linear in between.
      It stays below the decision threshold (0.5), so the dark area never marks
      an image as defective by itself - it only ranks the images that look OK.
    """
    shape_defect = (table["holes"] < NORMAL_HOLES) | (table["outline_edges"] > NORMAL_OUTLINE_EDGES)

    distance = (table["dark_fraction"] - DARK_TYPICAL).abs() / DARK_SPREAD
    dark_score = ((distance - DARK_Z_LOW) / (DARK_Z_HIGH - DARK_Z_LOW)).clip(0, 1) * DARK_SCORE_MAX

    return np.where(shape_defect, 1.0, dark_score).round(3)


def predict_defect(table):
    """Decision rule: suspected defect (1) if score >= threshold, else OK (0)."""
    return (defect_score(table) >= DECISION_THRESHOLD).astype(int)


dev = dev.assign(score=defect_score(dev), predicted_defect=predict_defect(dev))

print("\n" + "=" * 50)
print("Step B - split and decision rule")
print(f"Development set: {len(dev)} images ({dev['inspector_label'].sum()} labeled Defective)")
print(f"Validation set:  {len(val)} images ({val['inspector_label'].sum()} labeled Defective)"
      " - kept aside for Part 4")
print(f"\nRule: Defective if holes < {NORMAL_HOLES} or outline_edges > {NORMAL_OUTLINE_EDGES}")
print(f"Dark area (learned from development OK images): typical {DARK_TYPICAL * 100:.3f}%, "
      f"spread {DARK_SPREAD * 100:.3f}%")

print("\nDevelopment set - rule result by inspector_type:")
dev_check = pd.crosstab(dev["inspector_type"], dev["predicted_defect"].map({0: "Predicted OK", 1: "Predicted Defective"}))
print(dev_check)

caught = ((dev["inspector_label"] == 1) & (dev["predicted_defect"] == 1)).sum()
false_alarms = dev[(dev["inspector_label"] == 0) & (dev["predicted_defect"] == 1)]
print(f"\nDefects caught: {caught} of {dev['inspector_label'].sum()}")
print(f"False alarms on images labeled OK: {len(false_alarms)}")
print(false_alarms[["image_id", "holes", "outline_edges"]].to_string(index=False))

# How the dark area grades the images that the rule calls OK
predicted_ok = dev[dev["predicted_defect"] == 0]
print("\nDevelopment set - score of images predicted OK, by inspector_type:")
print(predicted_ok.groupby("inspector_type")["score"]
      .agg(count="count", above_zero=lambda s: (s > 0).sum(), median="median", max="max")
      .round(3))

# ---------------------------------------------------------------
# Step C: apply the same rule to test.csv and write predictions.csv
#   The rule is applied unchanged - test images are never labeled or tuned on.
# ---------------------------------------------------------------
PREDICTIONS_PATH = Path(__file__).parent / "predictions.csv"

predictions = pd.DataFrame({
    "image_id": test["image_id"],
    "predicted_defect": predict_defect(test),
    "score": defect_score(test),
})
predictions.to_csv(PREDICTIONS_PATH, index=False)

# Sanity checks on the output file
saved = pd.read_csv(PREDICTIONS_PATH)
assert list(saved.columns) == ["image_id", "predicted_defect", "score"], "wrong columns"
assert len(saved) == len(test) == 72, "expected one row per test image (72)"
assert saved["image_id"].is_unique, "duplicate image_id"
assert set(saved["image_id"]) == set(test["image_id"]), "image_id does not match test.csv"
assert saved["predicted_defect"].isin([0, 1]).all(), "predicted_defect must be 0 or 1"
assert saved["score"].between(0, 1).all(), "score must be between 0 and 1"

print("\n" + "=" * 50)
print("Step C - predictions for test.csv")
print(f"Saved to: {PREDICTIONS_PATH}")
print(f"Rows: {len(saved)} (all checks passed)")
print(f"Predicted Defective: {saved['predicted_defect'].sum()}, "
      f"Predicted OK: {(saved['predicted_defect'] == 0).sum()}")
print("\nFirst rows:")
print(saved.head(10).to_string(index=False))

MODEL_STRENGTHS_MESSAGE = (
    "In the picture my model missed, there is a visible scratch. My model is very good at detecting "
    "the bigger defects - missing holes or edge chips. It is not so good at detecting scratches - "
    "the scratch is similar in color to its surroundings, so my model misses it easily, except when "
    "it is on a hole or something darker, and the contrast finds it."
)
print("\nAbout the model's strengths and weaknesses:")
print(MODEL_STRENGTHS_MESSAGE)
show_message_and_wait("What my model detects well, and what it misses", MODEL_STRENGTHS_MESSAGE)

print("\n" + "#" * 50)
print("END OF PART 3, START OF PART 4")
print("#" * 50)

# ===============================================================
# PART 4 - Checking the results (validation set only)
# ===============================================================

# ---------------------------------------------------------------
# 1. Confusion matrix, Precision and Recall on the validation set
#    The validation set was not used to build the rule (Part 3),
#    so this is a fair estimate of how the rule works on new images.
# ---------------------------------------------------------------
val = val.assign(score=defect_score(val), predicted_defect=predict_defect(val))

actual = val["inspector_label"]
predicted = val["predicted_defect"]
tp = int(((actual == 1) & (predicted == 1)).sum())  # defective, caught
fn = int(((actual == 1) & (predicted == 0)).sum())  # defective, missed
fp = int(((actual == 0) & (predicted == 1)).sum())  # OK, false alarm
tn = int(((actual == 0) & (predicted == 0)).sum())  # OK, correctly passed

precision = tp / (tp + fp)  # of the images we flagged, how many are really defective
recall = tp / (tp + fn)  # of the really defective images, how many we flagged

confusion = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual OK", "Actual Defective"],
    columns=["Predicted OK", "Predicted Defective"],
)

print("\n" + "=" * 50)
print(f"Validation set: {len(val)} images ({int(actual.sum())} labeled Defective)")
print("\nConfusion matrix:")
print(confusion)
print(f"\nPrecision: {precision:.2f}  ({tp} of {tp + fp} flagged images are labeled Defective)")
print(f"Recall:    {recall:.2f}  ({tp} of {tp + fn} Defective images were flagged)")

print("\nValidation set - rule result by inspector_type:")
print(pd.crosstab(val["inspector_type"],
                  val["predicted_defect"].map({0: "Predicted OK", 1: "Predicted Defective"})))

val_false_alarms = val[(actual == 0) & (predicted == 1)]
print("\nFalse alarms (labeled OK, predicted Defective):")
if val_false_alarms.empty:
    print("  none")
else:
    print(val_false_alarms[["image_id", "holes", "outline_edges", "score"]].to_string(index=False))

print("\nMissed defects (labeled Defective, predicted OK):")
print(val[(actual == 1) & (predicted == 0)]
      [["image_id", "inspector_type", "camera", "exposure_setting", "holes", "outline_edges", "score"]]
      .to_string(index=False))

# Confusion matrix chart
fig_cm, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
ax.imshow(confusion.values, cmap="Blues")
for row in range(2):
    for col in range(2):
        value = confusion.values[row, col]
        text_color = "white" if value > confusion.values.max() / 2 else "#333333"
        ax.text(col, row, str(value), ha="center", va="center", fontsize=16, color=text_color)
ax.set_xticks([0, 1])
ax.set_xticklabels(confusion.columns)
ax.set_yticks([0, 1])
ax.set_yticklabels(confusion.index)
ax.set_title(f"Validation set: Precision {precision:.2f}, Recall {recall:.2f}")

CM_PATH = Path(__file__).parent / "confusion_matrix_validation.png"
fig_cm.savefig(CM_PATH, dpi=150)
plt.close(fig_cm)
print(f"\nConfusion matrix chart saved to: {CM_PATH}")
show_and_wait(CM_PATH, "Confusion matrix - validation set")

# ---------------------------------------------------------------
# 2. One image the rule got wrong, and a possible reason
# ---------------------------------------------------------------
ERROR_IMAGE_ID = "IMG_0110"  # clearly visible scratch, but the rule predicted OK

# Write the possible reason here (it is printed and shown under the image):
ERROR_REASON = (
    "My solution was wrong because the scratch is white, and it's very close to the color\n"
    "of the areas near the scratch, so it couldn't detect it."
)

error_row = val[val["image_id"] == ERROR_IMAGE_ID].iloc[0]
reason_text = ERROR_REASON if ERROR_REASON else "(Possible reason: to be written)"

print("\n" + "=" * 50)
print(f"Image the rule got wrong: {ERROR_IMAGE_ID}")
print(f"  Inspector label: {LABEL_NAMES[error_row['inspector_label']]} ({error_row['inspector_type']})")
print(f"  Prediction:      {LABEL_NAMES[error_row['predicted_defect']]} (score {error_row['score']:.3f})")
print(f"  Features:        holes={error_row['holes']}, outline_edges={error_row['outline_edges']}, "
      f"mean_brightness={error_row['mean_brightness']:.1f}")
print(f"  Camera / exposure: {error_row['camera']} / {error_row['exposure_setting']}")
print(f"  Reason: {' '.join(reason_text.split())}")  # one line in the terminal

fig_err, ax = plt.subplots(figsize=(6.5, 7.5), layout="constrained")
ax.imshow(plt.imread(IMAGES_DIR / error_row["file"]))
ax.set_title(
    f"{ERROR_IMAGE_ID} - labeled {LABEL_NAMES[error_row['inspector_label']]} "
    f"({error_row['inspector_type']}), predicted {LABEL_NAMES[error_row['predicted_defect']]}\n"
    f"holes={error_row['holes']}, outline edges={error_row['outline_edges']}, "
    f"score={error_row['score']:.3f}, {error_row['camera']}, exposure {error_row['exposure_setting']}",
    fontsize=10,
)
ax.set_xticks([])
ax.set_yticks([])
fig_err.supxlabel(reason_text, fontsize="medium")

ERROR_IMAGE_PATH = Path(__file__).parent / "validation_error_example.png"
fig_err.savefig(ERROR_IMAGE_PATH, dpi=150, bbox_inches="tight", pad_inches=0.2)
plt.close(fig_err)
print(f"\nError example saved to: {ERROR_IMAGE_PATH}")

MISSED_MESSAGE = "This is the defective component my model missed - a visible scratch."
print(MISSED_MESSAGE)
show_message_and_wait("The component my model missed", MISSED_MESSAGE)

show_and_wait(ERROR_IMAGE_PATH, f"Error example - {ERROR_IMAGE_ID}")

print("\n" + "#" * 50)
print("END OF PART 4, START OF PART 5")
print("#" * 50)

# ===============================================================
# PART 5 - Conclusions and use of AI
# ===============================================================
FINAL_CONCLUSIONS = (
    "1. Data quality: The day/night shift column did not match the capture time "
    "(82 of 168 rows were wrong), so I recomputed it from the timestamp (night = from 18:00).\n"
    "2. Batch C: Batch C has the highest rate of non-stain defects, both day and night. "
    "This points to a real quality problem in the components of this batch, "
    "not to the shooting conditions.\n"
    "3. Inspector labels are not perfect: 4 images labeled OK have a clearly visible defect "
    "(IMG_0092 edge chip; IMG_0112, IMG_0133, IMG_0167 missing hole). "
    "Several images labeled \"stain\" show no visible stain (e.g. IMG_0025).\n"
    "4. Shooting conditions: Some CAM-B images are very blurry or have a flash glare. "
    "Simple features like mean brightness or edge count do not separate good from defective "
    "components, because they are affected more by the camera than by the defect.\n"
    "5. My rule: On the validation set: Precision 1.00, Recall 0.71. It catches all missing holes "
    "and edge chips, but misses scratches and stains that do not cover a hole. The validation set "
    "is small (51 images), so these numbers are only an estimate.\n"
    "6. The rise in alerts: It is probably a mix of a real problem (Batch C), inconsistent labeling "
    "(stains), and image quality (blurry CAM-B images).\n"
    "Use of AI: I used Claude to help me write the code and to better present the data. "
    "The conclusions, decision making and ideas are all mine."
)

print("\nConclusions and use of AI:")
print(FINAL_CONCLUSIONS)
show_message_and_wait("Conclusions and use of AI", FINAL_CONCLUSIONS)

print("\n" + "#" * 50)
print("END OF PART 5")
print("#" * 50)
