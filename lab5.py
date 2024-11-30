import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean


# Load JSON data
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Calculate movement speeds
def calculate_speeds(data):
    speeds = defaultdict(list)
    for zebra, info in data.items():
        coords = info["gps coordinates"]
        timestamps = info["timestamp"]
        for i in range(1, len(coords)):
            distance = euclidean(
                [float(coords[i][0]), float(coords[i][1])],
                [float(coords[i - 1][0]), float(coords[i - 1][1])],
            )
            time_diff = float(timestamps[i]) - float(timestamps[i - 1])
            if time_diff > 0:
                speeds[zebra].append(distance / time_diff)
    return speeds


# Analyze zebra movement
def analyze_movement(speeds):
    return [speed for zebra_speeds in speeds.values() for speed in zebra_speeds]


# Plot CDF of movement speeds
def plot_speed_cdf(speeds):
    sorted_speeds = np.sort(speeds)
    cdf = np.arange(len(sorted_speeds)) / float(len(sorted_speeds))
    plt.figure()
    plt.plot(sorted_speeds, cdf, marker=".", linestyle="none")
    plt.xlabel("Speed (units per second)")
    plt.ylabel("CDF")
    plt.title("CDF of Zebra Movement Speeds")
    plt.grid()
    plt.show()


# Detect social behavior
def detect_social_behavior(data, distance_threshold=5):
    zebra_pairs = defaultdict(int)
    zebra_positions = {
        zebra: [list(map(float, coord)) for coord in info["gps coordinates"]]
        for zebra, info in data.items()
    }
    for zebra1, positions1 in zebra_positions.items():
        for zebra2, positions2 in zebra_positions.items():
            if zebra1 < zebra2:
                for p1, p2 in zip(positions1, positions2):
                    if euclidean(p1, p2) < distance_threshold:
                        zebra_pairs[(zebra1, zebra2)] += 1
    return {
        pair: count
        for pair, count in zebra_pairs.items()
        if count > len(data[zebra1]["gps coordinates"]) * 0.5
    }


# Plot heatmap of time spent
def plot_time_spent_heatmap(data, bin_size=0.5):
    location_counts = defaultdict(int)
    for zebra, info in data.items():
        coords = info["gps coordinates"]
        for coord in coords:
            binned_coord = (
                round(float(coord[0]) / bin_size) * bin_size,
                round(float(coord[1]) / bin_size) * bin_size,
            )
            location_counts[binned_coord] += 1

    x, y, counts = [], [], []
    for (coord_x, coord_y), count in location_counts.items():
        x.append(coord_x)
        y.append(coord_y)
        counts.append(count)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, weights=counts)
    plt.figure(figsize=(12, 10))
    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    plt.colorbar(label="Time Spent (instances)")
    plt.title("Heatmap of Time Spent by Zebras at Locations")
    plt.xlabel("Longitude (binned)")
    plt.ylabel("Latitude (binned)")
    plt.grid()
    plt.show()


# Detect sudden sound increases
def detect_sudden_sound_increases(data, sound_threshold=10):
    sudden_sound_events = []
    for zebra, info in data.items():
        sound_levels = info.get("sound levels", [])
        for i in range(1, len(sound_levels)):
            sound_increase = float(sound_levels[i]) - float(sound_levels[i - 1])
            if sound_increase > sound_threshold:
                sudden_sound_events.append((zebra, i, sound_increase))
    return sudden_sound_events


# Main execution
if __name__ == "__main__":
    file_path = "./simulation_2024_11_29_16_3_15_READABLE.json"
    data = load_data(file_path)

    speeds = calculate_speeds(data)
    all_speeds = analyze_movement(speeds)
    social_behavior = detect_social_behavior(data)

    print("Social Behavior Patterns:", social_behavior)

    plot_speed_cdf(all_speeds)
    plot_time_spent_heatmap(data)

    sudden_sound_events = detect_sudden_sound_increases(data, sound_threshold=10)
    print(
        "Sudden Sound Increases Detected (First 10 Events):", sudden_sound_events[:10]
    )
