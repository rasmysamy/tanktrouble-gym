import math

from pettingzoo import ParallelEnv

import functools
import random

from copy import copy
from math import sqrt

import matplotlib.pyplot as plt

from gymnasium.spaces import Discrete, MultiDiscrete, Box, Tuple, MultiBinary


class Ball:
    def __init__(self, x=-100, y=-100, vx=-100, vy=-100, life=-1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life

    def return_state(self):
        return self.x, self.y, self.vx, self.vy


class TankTrouble(ParallelEnv):
    metadata = {
        "name": "tanktrouble_v0",
    }

    def __init__(self, s_x=8, s_y=5):
        self.tank_length = 0.5
        self.tank_width = 0.3
        self.tank_speed = 1.25
        self.size_x = s_x
        self.size_y = s_y
        self.p1_x = None
        self.p1_y = None
        self.p2_x = None
        self.p2_y = None
        self.p1_direction = None
        self.p2_direction = None
        self.remaining_time = 100
        self.possible_agents = [0, 1]
        self.ball_speed = 2
        self.max_speed = 5
        self.p_wall = 0.4

        self.min_start_dist = max(self.size_x, self.size_y) / 1.33

        self.p1_vx = None
        self.p1_vy = None
        self.p2_vx = None
        self.p2_vy = None

        self.agent_name_mapping = {0: "p1", 1: "p2"}
        self.max_balls = 5
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]
        self.horizontal_walls = [[_ for _ in range(self.size_y + 1)] for _ in range(self.size_x)]
        self.vertical_walls = [[_ for _ in range(self.size_y)] for _ in range(self.size_x + 1)]
        self.action_spaces = [None, None]
        self.observation_spaces = [None, None]
        self.action_spaces[0] = MultiBinary(5)
        self.action_spaces[1] = MultiBinary(5)
        self.observation_spaces[0] = Tuple(
            [MultiBinary([self.size_x + 1, self.size_y]), MultiBinary([self.size_x, self.size_y + 1]),
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # self state
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # other state
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # own balls
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # other balls
             MultiBinary([self.max_balls]),  # own balls valid
             MultiBinary([self.max_balls]),  # other balls valid
             Box(low=0, high=self.remaining_time, shape=(1,))])
        self.observation_spaces[1] = Tuple(
            [MultiBinary([self.size_x + 1, self.size_y]), MultiBinary([self.size_x, self.size_y + 1]),
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # self state
             Box(low=min(-self.max_speed, 0), high=max(self.max_speed, self.size_x, self.size_y), shape=(4,)),
             # other state
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # own balls
             Box(low=0, high=max(self.size_y, self.size_x, self.ball_speed), shape=(4, self.max_balls)),  # other balls
             MultiBinary([self.max_balls]),  # own balls valid
             MultiBinary([self.max_balls]),  # other balls valid
             Box(low=0, high=self.remaining_time, shape=(1,))])

    def num_agents(self) -> int:
        return 2

    def max_num_agents(self) -> int:
        return 2

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def is_path(self, x1, y1, x2, y2):
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.floor(x2)
        y2 = math.floor(y2)
        # we do a bfs starting from x1, y1 and check if we can reach x2, y2
        visited = [[False for _ in range(self.size_y)] for _ in range(self.size_x)]
        queue = [(x1, y1)]
        visited[x1][y1] = True
        while len(queue) > 0:
            x, y = queue.pop(0)
            if x == x2 and y == y2:
                return True
            if x > 0 and not self.vertical_walls[x][y] and not visited[x - 1][y]:
                visited[x - 1][y] = True
                queue.append((x - 1, y))
            if x < self.size_x and not self.vertical_walls[x + 1][y] and not visited[x + 1][y]:
                visited[x + 1][y] = True
                queue.append((x + 1, y))
            if y > 0 and not self.horizontal_walls[x][y] and not visited[x][y - 1]:
                visited[x][y - 1] = True
                queue.append((x, y - 1))
            if y < self.size_y and not self.horizontal_walls[x][y + 1] and not visited[x][y + 1]:
                visited[x][y + 1] = True
                queue.append((x, y + 1))
        return False

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.size_x = 8
        self.size_y = 5

        while True:
            self.p1_x = random.randint(0, self.size_x - 1) + 0.5
            self.p1_y = random.randint(0, self.size_y - 1) + 0.5
            self.p2_x = random.randint(0, self.size_x - 1) + 0.5
            self.p2_y = random.randint(0, self.size_y - 1) + 0.5
            if sqrt((self.p1_x - self.p2_x) ** 2 + (self.p1_y - self.p2_y) ** 2) > self.min_start_dist:
                break

        self.p1_direction = random.random() * 2 * 3.14159
        self.p2_direction = random.random() * 2 * 3.14159
        self.p1_vx = 0
        self.p1_vy = 0
        self.p2_vx = 0
        self.p2_vy = 0
        self.remaining_time = 100
        self.p1_balls = [Ball() for _ in range(self.max_balls)]
        self.p2_balls = [Ball() for _ in range(self.max_balls)]

        while True:
            self.horizontal_walls = [[random.random() - (1 if idx in [0, self.size_y] else 0) < self.p_wall for idx in
                                      range(self.size_y + 1)] for _ in
                                     range(self.size_x)]  # outside walls are always there
            self.vertical_walls = [
                [random.random() - (1 if idx in [0, self.size_x] else 0) < self.p_wall for _ in range(self.size_y)] for
                idx in range(self.size_x + 1)]  # outside walls are always there
            # check if there is a path from p1 to p2
            if self.is_path(self.p1_x, self.p1_y, self.p2_x, self.p2_y):
                break

        print("reset")
        pass

    def display_state(self, show=True):
        # we draw the state using matplotlib
        plt.clf()
        plt.xlim(0, self.size_x)
        plt.ylim(0, self.size_y)
        self.draw_tank(self.p1_x, self.p1_y, self.p1_direction, 'r')
        self.draw_tank(self.p2_x, self.p2_y, self.p2_direction, 'b')
        plt.plot(self.p1_x, self.p1_y, 'ro')
        plt.plot(self.p2_x, self.p2_y, 'bo')
        plt.plot([self.p1_x, self.p1_x + 0.5 * math.cos(self.p1_direction)],
                 [self.p1_y, self.p1_y + 0.5 * math.sin(self.p1_direction)], 'r')
        plt.plot([self.p2_x, self.p2_x + 0.5 * math.cos(self.p2_direction)],
                 [self.p2_y, self.p2_y + 0.5 * math.sin(self.p2_direction)], 'b')
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.horizontal_walls[i][j]:
                    plt.plot([i, i + 1], [j, j], 'k')
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.vertical_walls[i][j]:
                    plt.plot([i, i], [j, j + 1], 'k')
        for ball in self.p1_balls:
            if ball.life > 0:
                plt.plot(ball.x, ball.y, 'r*')
        for ball in self.p2_balls:
            if ball.life > 0:
                plt.plot(ball.x, ball.y, 'b*')
        if show:
            plt.draw()
            plt.pause(1.0 / 30.0)
            plt.show()

    def draw_tank(self, x, y, direction, color):

        # draw a rectangle at x, y with the direction
        rect = plt.Rectangle((x - self.tank_length / 2, y - self.tank_width / 2), self.tank_length, self.tank_width,
                             angle=direction * 180 / 3.14159, color=color, rotation_point=(x, y))
        plt.gca().add_patch(rect)
        plt.plot([x, x + 0.5 * math.cos(direction)], [y, y + 0.5 * math.sin(direction)], color)

    def project(self, v1, v2):
        # project v1 onto v2
        return v1[0] * v2[0] + v1[1] * v2[1]

    def sat_test(self, points, obstacles, axis):
        # project all points onto the axis
        projections = [self.project(p, axis) for p in points]
        # project all obstacles onto the axis
        obstacle_projections = [self.project(p, axis) for p in obstacles]
        # check if the projections overlap
        return max(projections) >= min(obstacle_projections) and max(obstacle_projections) >= min(projections)

    def normalize(self, v):
        n = sqrt(v[0] * v[0] + v[1] * v[1])
        return [v[0] / n, v[1] / n]

    def perp_axis_from_points(self, p1, p2):
        # get the perpendicular axis to the line defined by p1 and p2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return self.normalize([-dy, dx])

    def player_segment_collision(self, px, py, rot, x1, y1, x2, y2):
        # first, we find the 4 corners of the tank
        c1x = px + (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c1y = py + (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c2x = px + (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c2y = py + (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c3x = px - (self.tank_length / 2) * math.cos(rot) + (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c3y = py - (self.tank_length / 2) * math.sin(rot) + (self.tank_width / 2) * math.sin(rot + math.pi / 2)
        c4x = px - (self.tank_length / 2) * math.cos(rot) - (self.tank_width / 2) * math.cos(rot + math.pi / 2)
        c4y = py - (self.tank_length / 2) * math.sin(rot) - (self.tank_width / 2) * math.sin(rot + math.pi / 2)

        c1 = [c1x, c1y]
        c2 = [c2x, c2y]
        c3 = [c3x, c3y]
        c4 = [c4x, c4y]

        # we can use the separating axis theorem to check if the player segment collides with the wall segment
        # lets first project onto the normal of the wall segment
        axis = self.perp_axis_from_points([x1, y1], [x2, y2])

        if not self.sat_test([c1, c2, c3, c4], [[x1, y1], [x2, y2]], axis):
            return False

        # now we project onto the normal of the player segment
        axis = self.perp_axis_from_points([c1x, c1y], [c2x, c2y])
        if not self.sat_test([[x1, y1], [x2, y2]], [c1, c2, c3, c4], axis):
            return False

        # now we project onto the player segment
        axis = [math.cos(rot), math.sin(rot)]
        if not self.sat_test([[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]], [[x1, y1], [x2, y2]], axis):
            return False

        return True




    def player_collision(self, px, py, rot):
        # check if
        pass
    def step(self, actions):


        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
