import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from collections import defaultdict


def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def find_mingle_locations(data, distance_threshold=5):
    """
    Identify locations where lions and zebras tend to mingle.

    Parameters:
        data (dict): Tracking data for both lions and zebras.
        distance_threshold (float): Maximum distance to consider mingling.

    Returns:
        set: Coordinates where mingling occurs.
    """
    lion_positions = {
        entity: [list(map(float, coord)) for coord in info["gps coordinates"]]
        for entity, info in data.items()
        if "Lion" in entity
    }
    zebra_positions = {
        entity: [list(map(float, coord)) for coord in info["gps coordinates"]]
        for entity, info in data.items()
        if "Zebra" in entity
    }

    mingle_locations = set()

    for lion, lion_coords in lion_positions.items():
        for zebra, zebra_coords in zebra_positions.items():
            for lion_coord, zebra_coord in zip(lion_coords, zebra_coords):
                if euclidean(lion_coord, zebra_coord) < distance_threshold:
                    mingle_locations.add(
                        (round(lion_coord[0], 2), round(lion_coord[1], 2))
                    )

    return mingle_locations


def plot_mingle_locations(mingle_locations):
    """
    Plot the locations where lions and zebras mingle.

    Parameters:
        mingle_locations (set): Set of coordinates where mingling occurs.
    """
    if not mingle_locations:
        print("No mingle locations found.")
        return

    x, y = zip(*mingle_locations)
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c="blue", alpha=0.6, label="Mingle Locations")
    plt.title("Locations Where Lions and Zebras Mingle")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    file_path = "./simulation_2024_11_29_16_3_15_READABLE.json"
    data = load_data(file_path)

    mingle_locations = find_mingle_locations(data, distance_threshold=5)
    print("Number of mingle locations found:", len(mingle_locations))

    plot_mingle_locations(mingle_locations)
