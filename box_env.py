import functools
import random
from copy import copy

import math
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from pettingzoo import ParallelEnv

import pygame
from Box2D import b2World, b2PolygonShape
import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)


class BoxAgent:
    def __init__(self, body: Box2D.b2Body):
        self.body = body
        self.yvels = []
        self.n = 20

    def step(self):
        self.yvels.append(abs(self.body.linearVelocity.y))
        if len(self.yvels) > self.n:
            self.yvels.pop(0)

    def can_jump(self):
        return len(self.yvels) > 0 and max(self.yvels) < 1


# class ContactDetector(contactListener):
#     def __init__(self, env):
#         contactListener.__init__(self)
#         self.env = env
#
#     def BeginContact(self, contact):
#         boxes = [b.body for b in self.env.boxes]
#         if contact.fixtureA.body in boxes and contact.fixtureB.body.is_grounded:
#             contact.fixtureA.body.is_grounded = True
#         elif contact.fixtureB.body in boxes:
#             contact.fixtureA.body.is_grounded = True
#             self.env.game_over = True
#         for i in range(2):
#             if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
#                 self.env.legs[i].ground_contact = True
#
#     def EndContact(self, contact):
#         for i in range(2):
#             if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
#                 self.env.legs[i].ground_contact = False


VIEWPORT_W = 600
VIEWPORT_H = 400
SCALE = 20  # Scale from pygame units to pixels
FPS = 40
MAX_HOR_SPD = 2

# H, W are in (Py)game units
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

FLOOR_Y = 15


class BoxJumpEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_boxes=4, world_width=10, world_height=6, box_width=1, box_height=1, render_mode=None,
                 gravity=10, friction=0.8, spacing=1.5, reward_scheme=1):
        self.num_boxes = num_boxes
        self.width = world_width
        self.height = world_height
        self.box_width = box_width
        self.box_height = box_height
        self.gravity = gravity
        self.friction = friction
        self.spacing = spacing
        self.reward_scheme = reward_scheme
        assert reward_scheme in [1, 2]

        low = [0, 0, -5, -5, -math.pi, -5, 0, 0, 0, 0, 0]
        high = [1, 1, 5, 5, math.pi, 5, 1, 1, 1, 1, 1]
        if self.reward_scheme == 2:
            low.append(0)
            high.append(1)
        size = len(low)
        low, high = np.array(low), np.array(high)
        self.obs_space = Box(low, high, shape=[size])

        # Construct state_space as a concatenation of subsets of all obs_spaces
        self.keep_inds = np.array([0, 1, 2, 3, 4, 10])
        low_state = np.repeat(low[self.keep_inds], num_boxes)
        high_state  = np.repeat(high[self.keep_inds], num_boxes)
        if self.reward_scheme == 2:
            low_state = np.concatenate((low_state, [0]))
            high_state = np.concatenate((high_state, [1]))
        size = low_state.shape[0]
        self.state_space = Box(low=low_state, high=high_state, shape=[size])

        self.boxes = None
        self.world = None
        self._state = None
        self.possible_agents = [f"box-{i}" for i in range(1, num_boxes+1)]

        self.render_mode = render_mode
        self.screen = None
        self.highest_y = -1

    @functools.lru_cache(maxsize=None)
    def action_space(self, _):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _):
        return self.obs_space

    def state(self):
        return self._state

    def get_all_obs(self):
        # obs =
        #  position x
        #  position y
        #  velocity x
        #  velocity y
        #  angle
        #  angular velocity
        #  is_standing_on_surface
        #  dist left (float)
        #  dist right (float)
        #  dist below (float)
        #  dist above (float)

        xs = np.array([box.body.transform.position.x for box in self.boxes])
        ys = np.array([box.body.transform.position.x for box in self.boxes])

        xs_i = np.argsort(xs)
        xs_s = xs[xs_i] / W
        left = np.concatenate(([1], xs_s[1:] - xs_s[:-1]))[xs_i]
        right = np.concatenate((xs_s[1:] - xs_s[:-1], [1]))[xs_i]

        # Todo: O(N^2) algo below could be replaced with a more efficient left-to-right sweep algorithm
        above = []
        below = []

        w = self.box_width / 2
        for i in range(self.num_boxes):
            # Extract boxes which overlap horizontally with this box
            x = xs[i]
            y = ys[i]
            relevant = np.logical_and(xs - w < x, xs + w > x)
            if np.sum(relevant) == 1:
                above.append(1)
                below.append((FLOOR_Y - y) / H)
            else:
                abv = np.logical_and(relevant, ys > y)
                blw = np.logical_and(relevant, ys < y)

                abv = ((y - np.max(ys[abv])) / H) if np.sum(abv)>0 else 1
                blw = ((np.min(ys[blw]) - y) / H) if np.sum(blw) > 0 else (FLOOR_Y - y) / H

                above.append(abv)
                below.append(blw)

        obs = {}
        state = []
        for i in range(self.num_boxes):
            b: BoxAgent = self.boxes[i]
            ob = [b.body.position.x / W,
                  b.body.position.y / H,
                  b.body.linearVelocity.x / W / FPS,
                  b.body.linearVelocity.y / H / FPS,
                  b.body.angle,
                  20 * b.body.angularVelocity / FPS,
                  left[i],   # distance to closest box on the left
                  right[i],  # distance to closest box on the right
                  above[i],  # distance to closest box above which overlaps horizontally
                  below[i],  # distance to closest box below which overlaps horizontally
                  int(b.can_jump())
            ]

            if self.reward_scheme == 2:
                ob.append(self.highest_y)

            ob = np.array(ob, dtype=np.float32)
            obs[self.agents[i]] = ob
            state.append(ob[self.keep_inds])

        if self.reward_scheme == 2:
            state.append(np.array([self.highest_y]))  # only include once
        self._state = np.concatenate(state)

        return obs

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.np_random = np.random.default_rng(seed)
        self._state = None
        self.world = b2World(gravity=(0, self.gravity), doSleep=True)
        self.highest_y = -1

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))

        # Make boxes
        self.boxes = []
        total_x = self.spacing * (self.num_boxes - 1)
        x = (W / 2) - (total_x / 2)
        start_y = FLOOR_Y - self.box_height / 2 - 1
        for i in range(self.num_boxes):
            body = self.world.CreateDynamicBody(position=(x, start_y))
            shape = b2PolygonShape(box=(self.box_width / 2, self.box_height / 2))
            body.CreateFixture(shape=shape, density=1, friction=self.friction)
            x += self.spacing
            body.angularDamping = 1

            body.ApplyForceToCenter((self.np_random.uniform(-150, 150), 0), True)

            self.boxes.append(BoxAgent(body))

        # Make floor
        floor_body = self.world.CreateStaticBody(position=(0, H))
        floor_shape = b2PolygonShape(box=(W, (H - FLOOR_Y)))
        floor_body.CreateFixture(shape=floor_shape)

        obs = self.get_all_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        rewards = {}

        prev_best = self.highest_y
        new_best = self.highest_y
        max_height_above_floor = -1

        for idx, i in enumerate(self.agents):
            action = actions[i]
            self.boxes[idx].step()
            body = self.boxes[idx].body
            if action == 0:
                pass
            elif action in [1, 2]:
                mul = -1 if action == 1 else 1
                if not ((mul > 0 and body.linearVelocity.x > MAX_HOR_SPD) or (mul < 0 and body.linearVelocity.x < -MAX_HOR_SPD)):
                    body.ApplyForceToCenter((mul * 10, 0), True)
            elif action == 3:
                if self.boxes[idx].can_jump():
                    body.ApplyForceToCenter((0, -250), True)

            height_above_floor = (FLOOR_Y - self.boxes[idx].body.position.y - self.box_height / 2) / FLOOR_Y
            if height_above_floor > new_best:
                new_best = height_above_floor
            max_height_above_floor = max(max_height_above_floor, height_above_floor)

        for i in self.agents:
            if self.reward_scheme == 1:
                rewards[i] = max_height_above_floor / self.num_boxes
            elif self.reward_scheme == 2:
                rewards[i] = (new_best - prev_best) / self.num_boxes

        self.world.Step(1 / FPS, 30, 30)
        self.timestep += 1

        terminations = {a: False for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 400:
            rewards = {a: 0 for a in self.agents}
            truncations = {a: True for a in self.agents}

        self.render()

        obs = self.get_all_obs()
        infos = {a: {} for a in self.agents}

        self.highest_y = new_best

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))

        square_image = pygame.Surface((self.box_width * SCALE, self.box_height * SCALE), pygame.SRCALPHA)
        square_image.fill((0, 0, 255))

        for b in self.boxes:
            position = b.body.position * SCALE
            angle = np.degrees(b.body.angle)

            rotated_square = pygame.transform.rotate(square_image, -angle)
            rotated_rect = rotated_square.get_rect(center=position)
            self.screen.blit(rotated_square, rotated_rect)

        # Render floor
        pygame.draw.rect(self.screen, (0, 0, 0), (0, FLOOR_Y * SCALE, W * SCALE, (H - FLOOR_Y) * SCALE))

        pygame.display.flip()


if __name__ == "__main__":
    # TEST()

    env = BoxJumpEnvironment(render_mode="human")

    n = 0
    obs, _ = env.reset(seed=n)
    import time

    while True:
        t = time.time()
        actions = {i: env.action_space(i).sample() for i in env.possible_agents}
        obs, rewards, term, trunc, info = env.step(actions)
        env.render()

        if not env.agents:
            n += 1
            obs, _ = env.reset(seed=n)

        length = time.time() - t
        if length < 1 / FPS:
            time.sleep(1 / FPS - length)

