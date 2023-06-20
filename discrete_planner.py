# encoding=utf-8
'''
Date: 2023-01-26 17:11:34
LastEditors: Lcf
LastEditTime: 2023-06-20 21:14:33
FilePath: \traj_planning\discrete_planner.py
Description: default
'''

import itertools
import os
import time
from queue import PriorityQueue

import addict
import matplotlib.pyplot as plt
import numpy as np
import yaml

# change working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working directory: {os.getcwd()}")

# load config
with open('planner_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    config = addict.Dict(config)  # convert to addict.Dict for easy access

# divide the configuration space into discrete grid
cell_size = config.min_turn_radius / 1.5

x_min, x_max = config.x_range
y_min, y_max = config.y_range

# the number of cells in x and y direction
num_cells_x_pos = np.ceil(x_max / cell_size).astype(int)
num_cells_x_neg = np.ceil(-x_min / cell_size).astype(int)
num_cells_y_pos = np.ceil(y_max / cell_size).astype(int)
num_cells_y_neg = np.ceil(-y_min / cell_size).astype(int)

num_cells_x = num_cells_x_pos + num_cells_x_neg
num_cells_y = num_cells_y_pos + num_cells_y_neg

print(f"Number of cells: x: {num_cells_x},\t y: {num_cells_y}")
print(f"cell size: {cell_size} m")


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


class Node:
    """
    Node class for A* search
    """

    delta = 0  # the tolerance for checking if two nodes are the same (speed up the search)

    def __init__(self, g, h, path, reached_waypoints):
        self.g = g  # cost (int)
        self.h = h  # heuristic (int)
        self.path = path  # list of index
        self.reached_waypoints = reached_waypoints  # list of waypoint index

    def __lt__(self, other):
        return False if self.__eq__(other) else self.g + self.h < other.g + other.h

    def __eq__(self, other):
        return other.g + other.h - self.delta <= self.g + self.h <= other.g + other.h + self.delta


def waypoints_kinematic_constraint(waypoints):
    # check if adjacent waypoints are the same
    for i in range(len(waypoints) - 1):
        if np.allclose(waypoints[i], waypoints[i + 1]):
            return False
    max_curvature = 1
    for i in range(len(waypoints) - 2):  # the curvature of the path should be smaller than the maximum curvature
        p1, p2, p3 = waypoints[i], waypoints[i + 1], waypoints[i + 2]
        a, b, c = np.linalg.norm(p2 - p3), np.linalg.norm(p1 - p3), np.linalg.norm(p1 - p2)
        if a + b < c and a + c < b and b + c < a:  # check if the three points are collinear
            sin_alpha = np.cross(p2 - p1, p3 - p1) / (b * c)
            curvature = 2 * sin_alpha / a
            if curvature > max_curvature:
                return False
        # else:
        #     # if p3 is between p1 and p2
        #     if np.dot(p3 - p1, p2 - p1) > 0 and np.dot(p3 - p2, p1 - p2) > 0:
        #
    return True

with Timer("Discret planning elapsed: %f s"):

    waypoints = np.array(config.waypoints)  # TODO
    # waypoints = np.random.randint(50 - 5, 50 + 5, (2, 2))
    # for i in range(len(waypoints)-1):
    #     # randomly generate 2 integers in the range of [0,1]
    #     a, b = np.random.randint(0, 2, 2)
    #     a = -1 if a == 0 else 1
    #     b = -1 if b == 0 else 1
    #     waypoints[i + 1, 0] = waypoints[i, 0] + a
    #     waypoints[i + 1, 1] = waypoints[i, 1] + b

    visited = np.zeros((num_cells_x, num_cells_y), dtype=int)
    def dist_infimum(p1, p2):
        """
        Calculate the infimum distance between two points
        """
        # return max(np.abs(p1[0] - p2[0]), np.abs(p1[1] - p2[1]))
        long_side = max(np.abs(p1[0] - p2[0]), np.abs(p1[1] - p2[1]))
        short_side = min(np.abs(p1[0] - p2[0]), np.abs(p1[1] - p2[1]))
        return long_side * 10 - short_side * 10 + short_side * 14


    # pre-compute the distance between waypoints to speed up the heuristic calculation
    wp_distance_lookup_table = np.zeros((len(waypoints)), dtype=float)
    for i in reversed(range(len(waypoints) - 1)):
        wp_distance_lookup_table[i] = wp_distance_lookup_table[i + 1] + \
                                      dist_infimum(waypoints[i], waypoints[i + 1])

    def heuristic(cell_index, waypoints, next_waypoint_index=0):
        """
        Calculate the heuristic cost between two cells
        """
        waypoints_h = wp_distance_lookup_table[next_waypoint_index]
        # current_h is the maximum of the distance between the current cell and the remaining waypoints
        current_h = dist_infimum(cell_index, waypoints[next_waypoint_index])
        return waypoints_h + current_h


    open_list = PriorityQueue()

    begin_cell_index = (50, 50)
    begin_direction = (1, 1)

    begin_cell_index_ = (begin_cell_index[0] + begin_direction[0], begin_cell_index[1] + begin_direction[1])

    # initialize the start node
    start_node = Node(g=0, h=heuristic(begin_cell_index, waypoints,
                                       next_waypoint_index=0), path=[begin_cell_index, begin_cell_index_], reached_waypoints=[])
    open_list.put(start_node)

    h = heuristic(begin_cell_index, waypoints, next_waypoint_index=0)

    # A* search
    while not open_list.empty():
        current_node = open_list.get()

        current_cell_index = current_node.path[-1]

        # check if the current cell is the goal cell
        if len(current_node.reached_waypoints) == len(waypoints):
            print(f"reached_waypoints: {current_node.reached_waypoints}")
            print("Found the goal cell!")
            print(f"Cost: {current_node.g}")
            print(f"Open list size: {open_list.qsize()}")
            break

        # expand the current cell
        for i, j in itertools.product(reversed(range(-1, 2)), reversed(range(-1, 2))):
            # skip the current cell
            if i == 0 and j == 0:
                continue

            # boundary check
            if current_cell_index[0] + i < 0 or current_cell_index[0] + i >= num_cells_x or \
                    current_cell_index[1] + j < 0 or current_cell_index[1] + j >= num_cells_y:
                continue

            # kinematic check
            if len(current_node.path) >= 2:  # not the start cell
                prev_direction = (current_node.path[-1][0] - current_node.path[-2][0],
                                  current_node.path[-1][1] - current_node.path[-2][1])
                current_direction = (i, j)
                # if dot product is negative, kinematic constraint is violated
                if prev_direction[0] * current_direction[0] + prev_direction[1] * current_direction[1] <= 0:
                    continue

            # calculate the cell index of next cell
            next_cell_index = (
                current_cell_index[0] + i, current_cell_index[1] + j)

            visited[next_cell_index] += 1

            next_waypoint_index = 0 if len(
                current_node.reached_waypoints) == 0 else min(current_node.reached_waypoints[-1] + 1,
                                                              len(waypoints) - 1)

            if np.array_equal(next_cell_index, waypoints[next_waypoint_index]):
                reached_waypoints = current_node.reached_waypoints + [next_waypoint_index]
            else:
                reached_waypoints = current_node.reached_waypoints

            dl = 10 if i == 0 or j == 0 else 14
            next_node = Node(g=current_node.g + dl, h=heuristic(next_cell_index, waypoints,
                                                                next_waypoint_index=next_waypoint_index),
                             path=current_node.path + [next_cell_index], reached_waypoints=reached_waypoints)

            open_list.put(next_node)


# plot the path
path = current_node.path
path_x = [cell_index[0] for cell_index in path]
path_y = [cell_index[1] for cell_index in path]

plt.figure()
plt.plot(path_x, path_y, 'r-')

# plot the waypoints
waypoints_x = [cell_index[0] for cell_index in waypoints]
waypoints_y = [cell_index[1] for cell_index in waypoints]
plt.plot(waypoints_x, waypoints_y, 'bo')

plt.axis('equal')

# plot the visited(numpy array) using plasma colormap
visited_x = np.where(visited)[0]
visited_y = np.where(visited)[1]
plt.scatter(visited_x, visited_y,
            c=visited[visited_x, visited_y], cmap='plasma', marker='o', s=1)
plt.colorbar()


from geomdl import fitting

degree = 4

with Timer("b-spline fitting elapsed: %f s"):

    # 1.remove points that are linear, keep only the turning points
    _path = np.array(path)
    for i in reversed(range(1, len(path) - 1)):
        if (path[i][0] - path[i - 1][0]) * (path[i + 1][1] - path[i][1]) == \
                (path[i][1] - path[i - 1][1]) * (path[i + 1][0] - path[i][0]):
            _path = np.delete(_path, i, axis=0)

    path = _path.tolist()

    # 2. linearly interpolate the last 2 points to make the path smoother
    last_point = path[-1]
    second_last_point = path[-2]
    # interpolate 10 points between the last 2 points
    path = path[:-2]  # remove the last 2 points
    for i in range(1, 10):
        path.append([second_last_point[0] + (last_point[0] - second_last_point[0]) / 10 * i,
                     second_last_point[1] + (last_point[1] - second_last_point[1]) / 10 * i])  # append the interpolated points
                    
    # 3. interpolate the path
    trajectory = fitting.interpolate_curve(path, degree)

    trajectory.delta = 0.01  # 100 sample points

# plot the trajectory
trajectory_x = [point[0] for point in trajectory.evalpts]
trajectory_y = [point[1] for point in trajectory.evalpts]

print(len(trajectory_x))

plt.plot(trajectory_x, trajectory_y, 'g-')

plt.title(f"Cost: {current_node.g}")

plt.show()
