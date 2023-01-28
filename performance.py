'''
Date: 2023-01-26 23:05:06
LastEditors: Lcf
LastEditTime: 2023-01-26 23:05:26
FilePath: \traj_planning\performance.py
Description: default
'''
from pyheat import PyHeat

ph = PyHeat('discrete_planner.py')
ph.create_heatmap()
ph.show_heatmap()