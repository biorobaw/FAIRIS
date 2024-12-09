import numpy as np
import json
import os 
os.chdir("../../..")
print(os.getcwd())

# Example adjusted points list (replace with your actual data)
adjusted_filtered_pose_list_1m = [
    (1.5, 2.0, 0.0),  # Example data points
    (2.0, 1.5, 1.57),
    (1.0, 2.5, 3.14)
]

# Generate the new pose list with (x, y) paired with 8 thetas (0 to 2π in π/4 intervals)
interval = np.pi / 4  # π/4 intervals for 8 thetas
thetas = [i * interval for i in range(8)]  # [0, π/4, ..., 7π/4]

# Create the updated pose list
expanded_pose_list = []
for (x, y, _) in adjusted_filtered_pose_list_1m:
    for theta in thetas:
        expanded_pose_list.append((x, y, theta))

# Save the updated pose list to a JSON file
file_path = "data/PathPoints/VPC2.json"
with open(file_path, "w") as f:
    json.dump(expanded_pose_list, f)

print(f"Expanded pose list saved to {file_path}")
