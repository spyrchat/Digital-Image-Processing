import numpy as np
import matplotlib.pyplot as plt

def find_four_closest_centers(points, known_point):
    points_array = np.array(points)
    diff = points_array - known_point
    squared_diff = diff ** 2
    squared_distances = np.sum(squared_diff, axis=1)
    closest_indices = np.argsort(squared_distances)[:4]
    closest_centers = [points[i] for i in closest_indices]
    return closest_centers

top_left_corners = []
for i in range(0, 360, 64):
    for j in range(0, 480, 48):
            top_left_corners.append((i, j))
centers = []
for i in top_left_corners:
    center_y = i[0] + 48 // 2
    center_x = i[1] + 64 //  2
    centers.append((center_y, center_x))
print("centers :", centers)
contexial_neighbours = find_four_closest_centers(centers, (125, 125))
print(contexial_neighbours)
w_left = 0
w_right = 0
h_up = 0
h_down = 0

for i in contexial_neighbours:
    if i[1] < 125 and (w_left == 0 or i[1] > w_left):
        w_left = i[1]
    if i[1] >= 125 and (w_right == 0 or i[1] < w_right):
        w_right = i[1]
    if i[0] > 125 and (h_up == 0 or i[0] < h_up):
        h_up = i[0]
    if i[0] <= 125 and (h_down == 0 or i[0] > h_down):
        h_down = i[0]

x_coords = [point[1] for point in contexial_neighbours]
y_coords = [point[0] for point in contexial_neighbours]
print(w_left, w_right, h_up, h_down)

plt.scatter(y_coords, x_coords, color='blue', label='Centers')  # Plot centers
plt.scatter(125, 125, color='red', label='Point (125,125)')  # Plot the specific point\
plt.gca().invert_yaxis()

plt.show()
