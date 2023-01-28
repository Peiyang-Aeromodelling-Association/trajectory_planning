import itertools
import numpy as np
from math import sqrt
from PIL import Image
import time


class A_star:
    def __init__(self, matrix, weights=1, corner_amend=1, step=float("inf"),
                 way=["R", "L", "D", "U", "RU", "RD", "LU", "LD"], wall=0):
        self.matrix = matrix
        self.weights = weights
        self.corner_amend = corner_amend
        self.matrix_length = len(self.matrix[0])
        self.matrix_width = len(self.matrix)
        self.step = step
        self.way = way
        self.wall = wall
        self.field = np.array(np.copy(self.matrix), dtype=float)
        for i, j in itertools.product(range(self.matrix_width), range(self.matrix_length)):
            if self.field[i][j] == self.wall:
                self.field[i][j] = float("inf")

    def run(self, start_point, end_point):
        self.fieldpointers = np.array(np.copy(self.matrix), dtype=str)
        self.start_point = start_point
        self.end_point = end_point
        if int(self.matrix[self.start_point[0]][self.start_point[1]]) == self.wall or int(
                self.matrix[self.end_point[0]][self.end_point[1]] == self.wall):
            exit("start or end is wall")
        self.fieldpointers[self.start_point[0]][self.start_point[1]] = "S"
        self.fieldpointers[self.end_point[0]][self.end_point[1]] = "G"
        return self.a_star()

    def a_star(self):
        setopen = [self.start_point]
        setopencosts = [0]
        setopenheuristics = [float("inf")]
        setclosed = []
        setclosedcosts = []
        movementdirections = self.way
        while self.end_point not in setopen and self.step:
            self.step -= 1
            total_costs = list(np.array(setopencosts) + self.weights * np.array(setopenheuristics))
            temp = np.min(total_costs)
            ii = total_costs.index(temp)
            if setopen[ii] != self.start_point and self.corner_amend == 1:
                new_ii = self.Path_optimization(temp, ii, setopen, setopencosts, setopenheuristics)
                ii = new_ii
            [costs, heuristics, posinds] = self.findFValue(setopen[ii], setopencosts[ii])
            setclosed = setclosed + [setopen[ii]]
            setclosedcosts = setclosedcosts + [setopencosts[ii]]
            setopen.pop(ii)
            setopencosts.pop(ii)
            setopenheuristics.pop(ii)
            for jj in range(len(posinds)):
                if float("Inf") != costs[jj]:
                    if posinds[jj] not in setclosed + setopen:
                        self.fieldpointers[posinds[jj][0]][posinds[jj][1]] = movementdirections[jj]
                        setopen = setopen + [posinds[jj]]
                        setopencosts = setopencosts + [costs[jj]]
                        setopenheuristics = setopenheuristics + [heuristics[jj]]
                    elif posinds[jj] in setopen:
                        position = setopen.index(posinds[jj])
                        if setopencosts[position] > costs[jj]:
                            setopencosts[position] = costs[jj]
                            setopenheuristics[position] = heuristics[jj]
                            self.fieldpointers[setopen[position][0]][setopen[position][1]] = movementdirections[jj]
                    else:
                        position = setclosed.index(posinds[jj])
                        if setclosedcosts[position] > costs[jj]:
                            setclosedcosts[position] = costs[jj]
                            self.fieldpointers[setclosed[position][0]][setclosed[position][1]] = movementdirections[jj]
            if not setopen:
                exit("Can't")
        if self.end_point in setopen:
            return self.findWayBack(self.end_point)
        else:
            exit("Can't")

    def Path_optimization(self, temp, ii, setOpen, setOpenCosts, setOpenHeuristics):
        [row, col] = setOpen[ii]
        _temp = self.fieldpointers[row][col]
        if "L" in _temp:
            col -= 1
        elif "R" in _temp:
            col += 1
        if "U" in _temp:
            row -= 1
        elif "D" in _temp:
            row += 1
        if [row, col] == self.start_point:
            new_ii = ii
        else:
            _temp = self.fieldpointers[row][col]
            [row2, col2] = [row, col]
            if "L" in _temp:
                col2 += self.matrix_width
            elif "R" in _temp:
                col2 -= self.matrix_width
            if "U" in _temp:
                row2 += 1
            elif "D" in _temp:
                row2 -= 1

            if 0 <= row2 <= self.matrix_width and 0 <= col2 <= self.matrix_length:
                new_ii = ii
            elif self.fieldpointers[setOpen[ii][0]][setOpen[ii][1]] == self.fieldpointers[row][col]:
                new_ii = ii
            elif [row2, col2] in setOpen:
                untext_ii = setOpen.index([row2, col2])
                now_cost = setOpenCosts[untext_ii] + self.weights * setOpenHeuristics[untext_ii]
                new_ii = untext_ii if temp == now_cost else ii
            else:
                new_ii = ii
        return new_ii

    def findFValue(self, currentpos, costsofar):
        cost = []
        heuristic = []
        posinds = []
        for way in self.way:
            if "D" in way:
                x = currentpos[0] - 1
            elif "U" in way:
                x = currentpos[0] + 1
            else:
                x = currentpos[0]
            if "R" in way:
                y = currentpos[1] - 1
            elif "L" in way:
                y = currentpos[1] + 1
            else:
                y = currentpos[1]
            if 0 <= y <= self.matrix_length - 1 and 0 <= x <= self.matrix_width - 1:
                posinds.append([x, y])
                heuristic.append(sqrt((self.end_point[1] - y) ** 2 + (self.end_point[0] - x) ** 2))
                cost.append(costsofar + self.field[x][y])
            else:
                posinds.append([0, 0])
                heuristic.append(float("inf"))
                cost.append(float("inf"))
        return [cost, heuristic, posinds]

    def findWayBack(self, goal):
        road = [goal]
        [x, y] = goal
        while self.fieldpointers[x][y] != "S":
            temp = self.fieldpointers[x][y]
            if "L" in temp:
                y -= 1
            if "R" in temp:
                y += 1
            if "U" in temp:
                x -= 1
            if "D" in temp:
                x += 1
            road.append([x, y])
        return road


if __name__ == "__main__":
    begintime = time.time()
    image = Image.open("blackwhite.bmp")
    # 如果本来就是黑白图片可以省去下列这条代码
    matrix = np.array(image.convert("1"))
    start_point = [3, 3]
    end_point = [110, 110]
    Road = A_star(matrix=matrix).run(start_point, end_point)
    for i in Road:
        image.putpixel((i[1], i[0]), (255, 0, 0))
    image.save("maze_reference.png")
    print(time.time() - begintime)