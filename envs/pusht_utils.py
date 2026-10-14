import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
import gymnasium
from gymnasium import spaces
from typing import Dict, Sequence, Union, Optional, Tuple
import os
import gdown
import zarr

# --- Pymunk Override ---
from pymunk.space_debug_draw_options import SpaceDebugColor

positive_y_is_up: bool = False

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)
        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])

def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

# --- Helper Functions ---

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
    assert(n_points >= 1)
    chosen_points = [points[init_idx]]
    for _ in range(n_points-1):
        cpoints = np.array(chosen_points)
        all_dists = np.linalg.norm(points[:,None,:] - cpoints[None,:,:], axis=-1)
        min_dists = all_dists.min(axis=1)
        next_idx = np.argmax(min_dists)
        next_pt = points[next_idx]
        chosen_points.append(next_pt)
    result = np.array(chosen_points)
    return result

# --- Keypoint Manager ---

from matplotlib import cm

class PymunkKeypointManager:
    def __init__(self, 
            local_keypoint_map: Dict[str, np.ndarray], 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        if color_map is None:
            try:
                cmap = cm.get_cmap('tab10')
            except AttributeError:  # matplotlib >= 3.9 removed cm.get_cmap
                import matplotlib
                cmap = matplotlib.colormaps['tab10']
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            'local_keypoint_map': self.local_keypoint_map,
            'color_map': self.color_map
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ['block','agent']:
            # Create a temporary simulation to extract keypoints
            space = pymunk.Space()
            if name == 'agent':
                obj = env.add_circle(space, (256, 400), 15)
                n_kps = n_agent_kps
            else:
                obj = env.add_tee(space, (256, 300), 0)
                n_kps = n_block_kps

            screen = pygame.Surface((512,512))
            screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(screen)
            space.debug_draw(draw_options)

            img = np.uint8(pygame.surfarray.array3d(screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255,255,255],dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512,512))), 0, -1)[:,:,::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1,2)).reshape(xy_img.shape)
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(
            translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(self, 
            pose_map: Dict[str, Union[Sequence, pymunk.Body]], 
            is_obj=False):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512,512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img


# --- Environments ---

class PushTEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            render_mode="rgb_array",
            rew_fn='sparse'
        ):
        self.render_mode = render_mode
        self._seed = None
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['render_fps']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.prev_action = None
        
        self.rew_fn = rew_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if self.prev_action is None:
            self.prev_action = action
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        done = coverage > self.success_threshold

        # Calculate reward based on rew_fn
        if self.rew_fn == 'sparse':
            reward = 0.0 if done else -1.0
        elif self.rew_fn == 'sparse_slow':
            reward = 0.0 if done else -1.0 * (1 + np.linalg.norm(action - self.prev_action)/200)
            self.prev_action = action
        elif self.rew_fn == 'l2':
            # L2 distance between block pose and goal pose
            block_pos = np.array(self.block.position)
            goal_pos = self.goal_pose[:2]
            pos_dist = np.linalg.norm(block_pos - goal_pos)

            # Angle distance (handle wraparound)
            block_angle = self.block.angle % (2 * np.pi)
            goal_angle = self.goal_pose[2] % (2 * np.pi)
            angle_diff = abs(block_angle - goal_angle)
            angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

            # Combine position and angle distances (normalize angle to similar scale)
            # Angle is weighted by a factor (e.g., 50 to match position scale)
            reward = -(pos_dist + 50 * angle_dist)
        elif self.rew_fn == 'l1':
            # L1 distance between block pose and goal pose
            block_pos = np.array(self.block.position)
            goal_pos = self.goal_pose[:2]
            pos_dist = np.sum(np.abs(block_pos - goal_pos))

            # Angle distance (handle wraparound)
            block_angle = self.block.angle % (2 * np.pi)
            goal_angle = self.goal_pose[2] % (2 * np.pi)
            angle_diff = abs(block_angle - goal_angle)
            angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

            # Combine position and angle distances (normalize angle to similar scale)
            reward = -(pos_dist + 50 * angle_dist)
        else:
            raise ValueError(f"Unknown reward function: {self.rew_fn}")

        terminated = bool(done)
        truncated = False # PushT usually doesn't have a time limit in the env itself, handled by wrapper

        observation = self._get_obs()
        info = self._get_info(coverage=coverage)
        info['success'] = 1.0 if done else 0.0
        info['reward'] = float(reward)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        return self._render_frame(self.render_mode)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self, coverage=None):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'coverage': coverage}
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                # cv2.drawMarker(img, coord,
                #     color=(255,0,0), markerType=cv2.MARKER_CROSS,
                #     markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        if self.legacy:
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle(self.space, (256, 400), 15)
        self.block = self.add_tee(self.space, (256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')
        return shape

    def add_circle(self, space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        space.add(body, shape)
        return body

    def add_tee(self, space, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        space.add(body, shape1, shape2)
        return body

class PushTKeypointsEnv(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            rew_fn='sparse',
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None,
            render_mode="rgb_array"
        ):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            rew_fn=rew_fn,
            render_action=render_action,
            render_mode=render_mode)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        rng = np.random.default_rng(self._seed)
        visible_kps = rng.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map

        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs


    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints and img is not None:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img

# --- Factory and Dataset ---

def make_env(env_name, seed=None, rew_fn='sparse', obstacle_cfg=None, subgoal_cfg=None):
    """Build a PushT env.

    With no obstacle (``obstacle_cfg`` None and no '-obstacle' in ``env_name``)
    this is exactly upstream's env. Otherwise a PushTObstacleWrapper is added:
    ``'pusht-keypoints-obstacle-v0'`` / ``'pusht-obstacle-v0'`` enable the
    paper's PushT-Obs obstacle (PUSHT_OBSTACLE_DEFAULTS), and ``obstacle_cfg``
    (or ``subgoal_cfg['obstacle']``) overrides individual keys. ``subgoal_cfg``
    with ``reward_mode == 'dense_parity'`` switches the scalar reward to the
    aggSAC reward (see PushTObstacleWrapper).
    """
    base_name, obstacle_cfg = resolve_obstacle_cfg(env_name, obstacle_cfg, subgoal_cfg)
    if obstacle_cfg is None and subgoal_cfg is not None:
        raise ValueError(f"{env_name!r} has no subgoal decomposition; use 'pusht-keypoints-obstacle-v0' "
                         "(or pass an obstacle config) for the multi-objective PushT-Obs task.")
    render_kwargs = {} if obstacle_cfg is None else dict(render_size=int(obstacle_cfg.get('render_size', 96)))
    if base_name == 'pusht-keypoints-v0':
        env = PushTKeypointsEnv(render_mode="rgb_array", rew_fn=rew_fn, **render_kwargs)
    elif base_name == 'pusht-v0':
        env = PushTEnv(render_mode="rgb_array", rew_fn=rew_fn, **render_kwargs)
    else:
        raise ValueError(f"Unknown env name: {env_name}")

    if seed is not None:
        env.reset(seed=seed)

    # Add TimeLimit wrapper (300 steps is standard for PushT)
    from gymnasium.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=300)

    # Obstacle wrapper (multi-objective PushT-Obs). It sits inside the
    # normalization wrapper so it reads pixel-space positions.
    if obstacle_cfg is not None:
        env = PushTObstacleWrapper(env, obstacle_cfg=obstacle_cfg)

    # Add Normalization wrapper
    env = PushTNormalizationWrapper(env)

    return env

class PushTNormalizationWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_val = 0.0
        self.max_val = 512.0

        # Update observation space
        low = np.full(self.observation_space.shape, -1.0, dtype=np.float32)
        high = np.full(self.observation_space.shape, 1.0, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

        # Update action space
        low = np.full(self.action_space.shape, -1.0, dtype=np.float32)
        high = np.full(self.action_space.shape, 1.0, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

    def normalize(self, x):
        return np.clip(2 * (x - self.min_val) / (self.max_val - self.min_val) - 1, -1.0, 1.0)

    def unnormalize(self, x):
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        action = self.unnormalize(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info

def _get_upstream_dataset(env_name):
    if env_name == 'pusht-keypoints-v0':
        dataset_path = os.path.join(os.path.expanduser('~/.ogpo/datasets/pusht'), 'pusht', 'pusht_cchi_v7_replay.zarr')
        if not os.path.exists(dataset_path):
            print(f"Downloading PushT dataset to {dataset_path}...")
            # Create directory
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            # Download zip
            url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
            zip_path = dataset_path + ".zip"
            import urllib.request
            urllib.request.urlretrieve(url, zip_path)
            # Unzip
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(os.path.dirname(dataset_path)))
            # Remove zip
            os.remove(zip_path)

        print(f"Loading dataset from {dataset_path}")
        import zarr
        root = zarr.open(dataset_path, mode='r')

        # Load data
        data = {
            'observations': root['data']['state'][:],
            'actions': root['data']['action'][:],
            'terminals': root['data']['n_contacts'][:] > 0, # Placeholder, will fix below
            'rewards': root['data']['n_contacts'][:] # Placeholder
        }

        # Fix terminals and rewards
        # The dataset doesn't have explicit terminals/rewards for RL, so we generate them
        # We need to run the environment to generate rewards and next_observations
        # Or we can just use the state transitions

        # Actually, for PushT, we can just use the state/action data directly
        # But we need next_observations and rewards for RL

        # Let's use the environment to generate proper RL transitions
        # This is a bit slow but ensures consistency

        # Optimization: The dataset contains state trajectories. 
        # We can just shift states to get next_observations.
        # Rewards can be calculated from coverage.

        # Re-loading to ensure we have clean data
        states = root['data']['state'][:]
        actions = root['data']['action'][:]
        episode_ends = root['meta']['episode_ends'][:]

        # Create lists
        observations = []
        next_observations = []
        act = []
        rew = []
        term = []
        masks = []

        # We need to generate keypoints from states
        env = make_env(env_name)
        env.reset()
        # Unwrap normalization for internal use if needed, but actually we want normalized data in dataset
        # So we should use the wrapped env to get normalized observations?
        # No, make_env now returns a normalized env.
        # But we need to be careful. The dataset contains raw states.
        # We should convert raw states to raw keypoints, THEN normalize.

        # Access the underlying env to get keypoint manager
        # The env is TimeLimit(PushTNormalizationWrapper(TimeLimit(PushTKeypointsEnv)))
        # Wait, make_env structure:
        # env = PushTKeypointsEnv()
        # env = TimeLimit(env)
        # env = PushTNormalizationWrapper(env)

        # So env.unwrapped is PushTKeypointsEnv
        kp_manager = env.unwrapped.kp_manager

        # Helper to normalize
        def normalize(x):
            return 2 * (x - 0.0) / (512.0 - 0.0) - 1

        print("Generating keypoints for dataset...")

        start_idx = 0
        for end_idx in episode_ends:
            episode_states = states[start_idx:end_idx]
            episode_actions = actions[start_idx:end_idx]

            # Generate keypoints for all states in episode
            # We can do this efficiently

            for i in range(len(episode_states)):
                state = episode_states[i]
                action = episode_actions[i]

                # Get observation (keypoints) from state
                # We need to set the state in the env to get the correct keypoints
                # Actually _set_state is fast enough.

                # We can use the static method if we refactor, but let's just use the env instance
                env.unwrapped._set_state(state)
                obs = env.unwrapped._get_obs()

                # Normalize observation
                obs = np.clip(normalize(obs), -1.0, 1.0)

                # Normalize action
                action = np.clip(normalize(action), -1.0, 1.0)

                observations.append(obs)
                act.append(action)

                # Next observation
                if i < len(episode_states) - 1:
                    next_state = episode_states[i+1]
                    env.unwrapped._set_state(next_state)
                    next_obs = env.unwrapped._get_obs()
                    next_obs = np.clip(normalize(next_obs), -1.0, 1.0)

                    # Reward (sparse)
                    # We can calculate coverage
                    # But simpler: -1 per step, 0 if success
                    # Success is coverage > 0.95
                    # We need to check coverage for next_state
                    # env._set_state updates the simulation, so we can check coverage
                    # But _set_state might not update coverage variable in env?
                    # _set_state calls _get_info which calculates coverage? No.
                    # Let's look at _get_info in PushTEnv

                    # We need to manually calculate coverage or step the env.
                    # Stepping is safer but slower.
                    # Given we have the trajectory, we assume the actions led to the next state.

                    # Let's just use -1 for all steps except the last one if it was successful?
                    # But some episodes might end early?
                    # Let's just use -1.0 for all transitions in dataset for now, 
                    # except maybe the last one?

                    reward = -1.0
                    done = False

                else:
                    # Last step
                    # Duplicate observation for next_obs (standard practice)
                    next_obs = obs.copy()
                    reward = 0.0 # Assume success at end of demonstration
                    done = True

                next_observations.append(next_obs)
                rew.append(reward)
                term.append(done)
                masks.append(1.0 - float(done))

            start_idx = end_idx

        data['observations'] = np.array(observations)
        data['actions'] = np.array(act)
        data['next_observations'] = np.array(next_observations)
        data['rewards'] = np.array(rew)
        data['terminals'] = np.array(term)
        data['masks'] = np.array(masks)

        return data
    else:
        raise ValueError(f"Unknown env name: {env_name}")


# =============================================================================
# Multi-objective PushT ("PushT-Obs", VaDaR paper appendix): a static obstacle,
# a sparse task signal and a continuous safety cost.
#
# Ported from the QIPM fork's pushT branch (OGPO@e08e82b, the code the paper's
# lambda sweep ran on). Nothing below changes upstream behavior: it is only
# active for '-obstacle' env names or when an obstacle config is passed.
# =============================================================================

import hashlib
import json
from dataclasses import dataclass

# PushTEnv.add_circle hardcodes the pusher radius to 15; keep in sync with PushTEnv._setup().
_PUSHER_RADIUS = 15.0

# The paper's PushT-Obs configuration (pushT-branch launchers / sweep scripts).
PUSHT_OBSTACLE_DEFAULTS = dict(
    enabled=True,
    shape='circle',            # 'circle' (x, y, r) or 'rect' (x, y, w, h); pixel coords in [0, 512]^2
    x=320.0,
    y=220.0,
    r=30.0,
    collide_with_pusher=True,  # the pusher disk (radius 15) counts for the cost / collisions
    collide_with_block=True,   # the T polygon counts for the cost / collisions
    terminate_on_collision=False,
    coverage_threshold=0.95,
    safe_margin=10.0,          # the cost decays linearly from 1 to 0 over this many pixels
    sparse_safety=False,       # ablation: binarize the cost to {0, 1}
    fixed_start_seed=None,     # if set, every reset uses this seed (a single layout)
    render_size=384,
    viz_seed=12345,
    # Scalar-reward (aggSAC) settings; used only when a dense reward mode is on.
    #   r = (dense_task_bonus if task_done else dense_sparse_step_cost)
    #       + dense_safety_coeff * (-dense_safety_scale * cost)
    dense_reward_mode='none',  # 'none' | 'parity' (aggSAC) | 'pbrs'
    dense_safety_coeff=1.0,    # lambda_safety
    dense_safety_scale=5.0,
    dense_task_bonus=15.0,     # the paper's runs used 15 (= VaDaR's task-head cap c_clip * r_done)
    dense_sparse_step_cost=-1.0,
    dense_pbrs_coeff=10.0,
    dense_discount=0.95,
)

_OBSTACLE_SUFFIX = '-obstacle'


def resolve_obstacle_cfg(env_name, obstacle_cfg=None, subgoal_cfg=None):
    """Return (base_env_name, obstacle_cfg or None) for `env_name`.

    'pusht-keypoints-obstacle-v0' -> ('pusht-keypoints-v0', paper defaults + overrides).
    A plain name with an enabled obstacle_cfg also gets the obstacle (QIPM-fork style).
    subgoal_cfg['obstacle'] is used when obstacle_cfg is None, and
    subgoal_cfg['reward_mode'] == 'dense_parity' selects the aggSAC reward.
    """
    base_name = env_name.replace(_OBSTACLE_SUFFIX, '')
    named_obstacle = base_name != env_name
    if obstacle_cfg is None and subgoal_cfg is not None:
        obstacle_cfg = subgoal_cfg.get('obstacle')
    if obstacle_cfg is not None:
        obstacle_cfg = dict(obstacle_cfg)
        if not obstacle_cfg.get('enabled', True):
            obstacle_cfg = None
    if obstacle_cfg is None and not named_obstacle:
        return base_name, None

    cfg = dict(PUSHT_OBSTACLE_DEFAULTS)
    cfg.update({k: v for k, v in (obstacle_cfg or {}).items() if v is not None})
    cfg['enabled'] = True
    if subgoal_cfg is not None:
        num = subgoal_cfg.get('num_subgoals')
        if num is not None and int(num) != 2:
            raise ValueError(f"PushT-Obs has 2 heads (task, cost); got num_subgoals={num}")
        mode = subgoal_cfg.get('reward_mode') or 'sparse'
        if mode == 'dense_parity':
            cfg['dense_reward_mode'] = 'parity'
        elif mode != 'sparse':
            raise ValueError(f"reward_mode {mode!r} is not supported for PushT-Obs (use 'sparse' or 'dense_parity')")
    return base_name, cfg


@dataclass
class CircleObstacle:
    """Circular obstacle in PushT pixel coordinates: center (x, y), radius r."""
    x: float
    y: float
    r: float

    @property
    def x_min(self) -> float:
        return self.x - self.r

    @property
    def y_min(self) -> float:
        return self.y - self.r

    @property
    def x_max(self) -> float:
        return self.x + self.r

    @property
    def y_max(self) -> float:
        return self.y + self.r

    def contains(self, px: float, py: float) -> bool:
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.r ** 2

    def signed_distance(self, px: float, py: float) -> float:
        """Positive outside, zero on the boundary, negative inside."""
        return ((px - self.x) ** 2 + (py - self.y) ** 2) ** 0.5 - self.r

    def polygon(self):
        return sg.Point(self.x, self.y).buffer(self.r)


@dataclass
class RectObstacle:
    """Axis-aligned rectangle obstacle in PushT pixel coordinates: corner (x, y), size (w, h)."""
    x: float
    y: float
    w: float
    h: float

    @property
    def x_min(self) -> float:
        return self.x

    @property
    def y_min(self) -> float:
        return self.y

    @property
    def x_max(self) -> float:
        return self.x + self.w

    @property
    def y_max(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float) -> bool:
        return (self.x_min <= px <= self.x_max) and (self.y_min <= py <= self.y_max)

    def signed_distance(self, px: float, py: float) -> float:
        """Euclidean distance to the rectangle outside; minus the penetration depth inside."""
        dx = max(self.x_min - px, 0.0, px - self.x_max)
        dy = max(self.y_min - py, 0.0, py - self.y_max)
        if dx == 0.0 and dy == 0.0:
            return -min(px - self.x_min, self.x_max - px, py - self.y_min, self.y_max - py)
        return (dx * dx + dy * dy) ** 0.5

    def polygon(self):
        return sg.box(self.x_min, self.y_min, self.x_max, self.y_max)


def make_obstacle(cfg):
    if str(cfg.get('shape', 'circle')).lower() == 'circle':
        return CircleObstacle(x=float(cfg['x']), y=float(cfg['y']),
                              r=float(cfg.get('r', cfg.get('radius', 30))))
    return RectObstacle(x=float(cfg['x']), y=float(cfg['y']), w=float(cfg['w']), h=float(cfg['h']))


class ObstacleCost:
    """Soft-margin proximity cost c in [0, 1], using full body geometry.

    c = 1 if the pusher disk (radius 15) or any part of the T polygon touches or
    penetrates the obstacle. Otherwise c = (m - d) / m for a clearance d < m =
    safe_margin (pusher-edge-to-obstacle or T-to-obstacle distance), and 0
    beyond the margin. The worst (largest) of pusher and block is returned.
    `sparse_safety` binarizes c to 1[c >= 1].
    """

    def __init__(self, cfg):
        self.obstacle = make_obstacle(cfg)
        self.obstacle_poly = self.obstacle.polygon()
        self.safe_margin = float(cfg.get('safe_margin', 10.0))
        self.collide_with_pusher = bool(cfg.get('collide_with_pusher', True))
        self.collide_with_block = bool(cfg.get('collide_with_block', True))
        self.sparse_safety = bool(cfg.get('sparse_safety', False))

    def _soft(self, clearance):
        if clearance <= 0.0:
            return 1.0
        if clearance < self.safe_margin:
            return (self.safe_margin - clearance) / self.safe_margin
        return 0.0

    def __call__(self, pusher_xy, block_body) -> float:
        worst = 0.0
        if self.collide_with_pusher:
            clearance = self.obstacle.signed_distance(float(pusher_xy[0]), float(pusher_xy[1])) - _PUSHER_RADIUS
            worst = max(worst, self._soft(clearance))
        if self.collide_with_block and worst < 1.0:
            t_geom = pymunk_to_shapely(block_body, block_body.shapes)
            if t_geom.intersects(self.obstacle_poly):
                worst = 1.0
            else:
                worst = max(worst, self._soft(float(t_geom.distance(self.obstacle_poly))))
        if self.sparse_safety:
            worst = 1.0 if worst >= 1.0 else 0.0
        return float(worst)


def _block_coverage(base_env) -> float:
    """Block-goal coverage of a PushTEnv, computed exactly as in PushTEnv.step."""
    goal_body = base_env._get_goal_pose_body(base_env.goal_pose)
    goal_geom = pymunk_to_shapely(goal_body, base_env.block.shapes)
    block_geom = pymunk_to_shapely(base_env.block, base_env.block.shapes)
    return goal_geom.intersection(block_geom).area / goal_geom.area


class PushTObstacleWrapper(gymnasium.Wrapper):
    """PushT with a static obstacle and two signals per step (PushT-Obs).

    The obstacle is not a physical body: nothing blocks the pusher or the T.
    A "collision" is geometric overlap (cost == 1).

    Per step:
        task_done = 1[coverage >= coverage_threshold] (ratchets within the episode, and the
                    episode terminates when it fires)
        cost      = ObstacleCost(pusher, T) in [0, 1]
        info['subgoal_dones'] = float32 [task_done, cost]  (VaDaR head 0 = task, head 1 = cost;
                    the cost head's target is -5 * cost + gamma^H * mask * Q')
        info['success'] = task_done and no collision (cost == 1) at any step of the episode
        info['naive_success'] = task_done; info['obstacle_avoidance'] = 1 - ever_collided
    Reward:
        dense_reward_mode 'none'   -> the base env reward (rew_fn), unchanged
        dense_reward_mode 'parity' -> aggSAC: (dense_task_bonus if task_done else dense_sparse_step_cost)
                                      + dense_safety_coeff * (-dense_safety_scale * cost)
        dense_reward_mode 'pbrs'   -> dense_discount * Phi(s') - Phi(s) + (dense_task_bonus if task_done)
                                      + dense_safety_coeff * (-dense_safety_scale * cost),
                                      Phi = dense_pbrs_coeff * coverage
    Reset: the usual random PushT start, re-drawn with seed, seed+1, ... until
    neither the pusher nor the T touches the obstacle; or always
    `fixed_start_seed` if that is set.

    Sits inside PushTNormalizationWrapper (it reads pixel-space positions).
    """

    def __init__(self, env, obstacle_cfg):
        super().__init__(env)
        cfg = dict(PUSHT_OBSTACLE_DEFAULTS)
        cfg.update(obstacle_cfg)
        self.obstacle_cfg = cfg
        self._cost_fn = ObstacleCost(cfg)
        self._obstacle = self._cost_fn.obstacle
        self._coverage_threshold = float(cfg['coverage_threshold'])
        self._terminate_on_collision = bool(cfg['terminate_on_collision'])

        self._dense_reward_mode = str(cfg['dense_reward_mode']).lower()
        if self._dense_reward_mode not in ('none', 'parity', 'pbrs'):
            raise ValueError(f"dense_reward_mode must be 'none' | 'parity' | 'pbrs', got {self._dense_reward_mode!r}")
        self._dense_safety_coeff = float(cfg['dense_safety_coeff'])
        self._dense_safety_scale = float(cfg['dense_safety_scale'])
        self._dense_task_bonus = float(cfg['dense_task_bonus'])
        self._dense_sparse_step_cost = float(cfg['dense_sparse_step_cost'])
        self._dense_pbrs_coeff = float(cfg['dense_pbrs_coeff'])
        self._dense_discount = float(cfg['dense_discount'])

        fss = cfg.get('fixed_start_seed')
        self._fixed_start_seed = int(fss) if fss is not None and fss != '' else None

        self._task_done = False
        self._ever_collided = False
        self._last_coverage = 0.0

    def _start_inside_obstacle(self) -> bool:
        """True if the pusher disk or the T touches the obstacle (unbinarized cost == 1)."""
        base = self.env.unwrapped
        sparse, self._cost_fn.sparse_safety = self._cost_fn.sparse_safety, False
        try:
            return self._cost_fn(base.agent.position, base.block) >= 1.0
        finally:
            self._cost_fn.sparse_safety = sparse

    def reset(self, **kwargs):
        seed = kwargs.pop('seed', None)
        if self._fixed_start_seed is not None:
            obs, info = self.env.reset(seed=self._fixed_start_seed, **kwargs)
            if self._start_inside_obstacle():
                raise RuntimeError(f'PushTObstacleWrapper: fixed_start_seed={self._fixed_start_seed} '
                                   'places the pusher/block inside the obstacle. Pick a different seed.')
        else:
            max_attempts = 100
            for attempt in range(max_attempts):
                s = seed + attempt if seed is not None else None
                obs, info = self.env.reset(seed=s, **kwargs)
                if not self._start_inside_obstacle():
                    break
            else:
                raise RuntimeError(f'PushTObstacleWrapper: no valid start pose (outside the obstacle) '
                                   f'after {max_attempts} attempts; starting seed={seed}')

        self._task_done = False
        self._ever_collided = False
        self._last_coverage = 0.0
        info = dict(info)
        info['subgoal_dones'] = np.zeros(2, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        base = self.env.unwrapped

        coverage = float(info.get('coverage', 0.0) or 0.0)
        if coverage >= self._coverage_threshold:
            self._task_done = True
        cost = self._cost_fn(base.agent.position, base.block)
        collided = cost >= 1.0
        self._ever_collided = self._ever_collided or collided
        task_done = self._task_done

        dense_task_term = dense_shaping_term = dense_safety_term = 0.0
        if self._dense_reward_mode == 'parity':
            dense_task_term = self._dense_task_bonus if task_done else self._dense_sparse_step_cost
            dense_safety_term = -self._dense_safety_scale * cost
            reward = dense_task_term + self._dense_safety_coeff * dense_safety_term
        elif self._dense_reward_mode == 'pbrs':
            dense_shaping_term = (self._dense_discount * self._dense_pbrs_coeff * coverage
                                  - self._dense_pbrs_coeff * self._last_coverage)
            dense_task_term = self._dense_task_bonus if task_done else 0.0
            dense_safety_term = -self._dense_safety_scale * cost
            reward = dense_shaping_term + dense_task_term + self._dense_safety_coeff * dense_safety_term
        self._last_coverage = coverage

        info['subgoal_dones'] = np.array([float(task_done), cost], dtype=np.float32)
        info['proximity_cost'] = cost
        info['ever_collided'] = float(self._ever_collided)
        info['success'] = 1.0 if (task_done and not self._ever_collided) else 0.0
        info['naive_success'] = 1.0 if task_done else 0.0
        info['obstacle_avoidance'] = 0.0 if self._ever_collided else 1.0
        if self._dense_reward_mode != 'none':
            info['dense_task_term'] = float(dense_task_term)
            info['dense_shaping_term'] = float(dense_shaping_term)
            info['dense_safety_term'] = float(dense_safety_term)
        info['reward'] = float(reward)

        if task_done:
            terminated = True
        if self._terminate_on_collision and collided:
            terminated = True
        return obs, float(reward), terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame
        try:
            ws = float(self.env.unwrapped.window_size)
            sx, sy = frame.shape[1] / ws, frame.shape[0] / ws
            out = frame.copy()
            overlay = out.copy()
            if isinstance(self._obstacle, CircleObstacle):
                c = (int(round(self._obstacle.x * sx)), int(round(self._obstacle.y * sy)))
                rr = int(round(self._obstacle.r * sx))
                cv2.circle(overlay, c, rr, (255, 0, 0), thickness=-1)
                cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
                cv2.circle(out, c, rr, (255, 0, 0), thickness=2)
            else:
                p0 = (int(round(self._obstacle.x_min * sx)), int(round(self._obstacle.y_min * sy)))
                p1 = (int(round(self._obstacle.x_max * sx)), int(round(self._obstacle.y_max * sy)))
                cv2.rectangle(overlay, p0, p1, (255, 0, 0), thickness=-1)
                cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)
                cv2.rectangle(out, p0, p1, (255, 0, 0), thickness=2)
            return out
        except Exception:
            return frame


def make_pusht_obstacle_video(agent, env, save_path, max_steps=300, seed=12345):
    """Roll out one episode of `agent` in an obstacle env and save a GIF with the obstacle drawn."""
    import imageio
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    obs, _ = env.reset(seed=seed)
    frames = []
    f0 = env.render()
    if f0 is not None:
        frames.append(f0)
    actor_fn = (getattr(agent, 'compute_flow_actions_corrected', None)
                or getattr(agent, 'compute_flow_actions', None) or agent.sample_actions)
    action_dim = env.action_space.shape[0]
    queue = []
    for _ in range(max_steps):
        if not queue:
            try:
                chunk = actor_fn(observations=obs[None])
            except TypeError:
                chunk = actor_fn(obs[None])
            queue = list(np.array(chunk).reshape(-1, action_dim))
        obs, _, terminated, truncated, _ = env.step(queue.pop(0))
        f = env.render()
        if f is not None:
            frames.append(f)
        if terminated or truncated:
            break
    if frames:
        imageio.mimsave(save_path, frames, duration=0.05)
    return save_path


# ---------------------------------------------------------------------------
# Offline datasets with subgoal_dones = [task_done, cost] (post-action labels).
# ---------------------------------------------------------------------------

_LABEL_CFG_KEYS = ('shape', 'x', 'y', 'r', 'w', 'h', 'coverage_threshold', 'safe_margin',
                   'collide_with_pusher', 'collide_with_block', 'sparse_safety')


def _pusht_label_poses(agent_xy, block_pose, obstacle_cfg):
    """(N, 2) float32 [coverage >= threshold, cost] for raw pixel poses.

    agent_xy: (N, 2) pusher positions; block_pose: (N, 3) [x, y, angle] of the T body.
    Uses the same geometry code as the live env (PushTEnv + ObstacleCost).
    """
    base = PushTEnv(render_mode="rgb_array")
    base.reset(seed=0)
    cost_fn = ObstacleCost(obstacle_cfg)
    thr = float(obstacle_cfg.get('coverage_threshold', 0.95))
    labels = np.zeros((len(agent_xy), 2), dtype=np.float32)
    for i in range(len(agent_xy)):
        base.agent.position = (float(agent_xy[i, 0]), float(agent_xy[i, 1]))
        base.block.angle = float(block_pose[i, 2])
        base.block.position = (float(block_pose[i, 0]), float(block_pose[i, 1]))
        labels[i, 0] = float(_block_coverage(base) >= thr)
        labels[i, 1] = cost_fn(base.agent.position, base.block)
    base.close()
    return labels


def _poses_from_observations(obs):
    """Recover (agent_xy, block_pose) in pixels from normalized PushT observations.

    5-D state obs: [agent_x, agent_y, block_x, block_y, angle], each mapped [0, 512] -> [-1, 1].
    40-D keypoint obs: 9 block keypoints + agent xy (+ 20 visibility flags); the T pose is the
    rigid 2-D Procrustes fit of the env's local block keypoints to the observed ones.
    """
    raw = (np.asarray(obs, dtype=np.float64) + 1.0) / 2.0 * 512.0
    if raw.shape[-1] == 5:
        return raw[:, :2], raw[:, 2:5]
    if raw.shape[-1] != 40:
        raise ValueError(f"Cannot recover PushT poses from observations of dim {raw.shape[-1]}")
    kp_local = np.asarray(
        PushTKeypointsEnv.genenerate_keypoint_manager_params()['local_keypoint_map']['block'], dtype=np.float64)
    n_kp = kp_local.shape[0]
    kp_global = raw[:, :2 * n_kp].reshape(-1, n_kp, 2)
    agent_xy = raw[:, 2 * n_kp:2 * n_kp + 2]
    mu_l = kp_local.mean(axis=0)
    mu_g = kp_global.mean(axis=1)
    L = kp_local - mu_l
    Gc = kp_global - mu_g[:, None, :]
    num = (L[None, :, 0] * Gc[..., 1] - L[None, :, 1] * Gc[..., 0]).sum(axis=1)
    den = (L[None, :, 0] * Gc[..., 0] + L[None, :, 1] * Gc[..., 1]).sum(axis=1)
    angle = np.arctan2(num, den)
    c, s = np.cos(angle), np.sin(angle)
    pos = mu_g - np.stack([c * mu_l[0] - s * mu_l[1], s * mu_l[0] + c * mu_l[1]], axis=-1)
    return agent_xy, np.concatenate([pos, angle[:, None]], axis=-1)


def _cumulative_task_done(labels, terminals):
    """Ratchet the task flag within episodes (an episode ends where terminals == 1)."""
    out = labels.copy()
    done = 0.0
    for i in range(len(out)):
        done = max(done, out[i, 0])
        out[i, 0] = done
        if terminals[i] > 0.5:
            done = 0.0
    return out


def _cached_labels(source_key, obstacle_cfg, compute_fn):
    cfg = {k: obstacle_cfg[k] for k in _LABEL_CFG_KEYS if k in obstacle_cfg}
    key = json.dumps({'src': source_key, 'cfg': cfg, 'v': 1}, sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = os.path.join(os.path.expanduser('~/.vadar/cache/pusht_subgoal_dones'), f'{digest}.npy')
    if os.path.exists(path):
        return np.load(path)
    labels = compute_fn()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, labels)
    except OSError as e:
        print(f"[pusht] could not cache subgoal labels at {path}: {e}")
    return labels


def _file_key(path):
    st_ = os.stat(path)
    return f'{os.path.abspath(path)}:{st_.st_size}:{int(st_.st_mtime)}'


def _upstream_subgoal_labels(base_name, terminals, obstacle_cfg):
    """Post-action labels for the upstream zarr demos, from the stored simulator states."""
    dataset_path = os.path.join(os.path.expanduser('~/.ogpo/datasets/pusht'), 'pusht', 'pusht_cchi_v7_replay.zarr')

    def compute():
        root = zarr.open(dataset_path, mode='r')
        states = root['data']['state'][:]
        episode_ends = root['meta']['episode_ends'][:]
        nxt = np.empty_like(states)
        start = 0
        for end in episode_ends:
            nxt[start:end - 1] = states[start + 1:end]
            nxt[end - 1] = states[end - 1]
            start = end
        labels = _pusht_label_poses(nxt[:, :2], nxt[:, 2:5], obstacle_cfg)
        return _cumulative_task_done(labels, terminals)

    return _cached_labels(f'{base_name}:{_file_key(dataset_path)}', obstacle_cfg, compute)


def load_bc_dataset(bc_dataset_path):
    """Load a BC npz (scripts/pusht/collect_bc_data.py) as a dict of arrays."""
    with np.load(os.path.expanduser(bc_dataset_path)) as f:
        return {k: f[k] for k in f.files}


def get_dataset(env_name, bc_dataset_path=None, obstacle_cfg=None, subgoal_cfg=None):
    """Offline PushT dataset.

    Without an obstacle and with no bc_dataset_path this is upstream's dataset
    (the diffusion-policy pusht_cchi_v7 demos). `bc_dataset_path` loads an npz
    of policy rollouts instead (the paper's PushT-Obs runs BC-pretrain on
    sac_pusht_2000000_bc.npz, collected from a vanilla SAC policy without the
    obstacle).

    With subgoal tracking (subgoal_cfg is not None) on the obstacle task the
    dataset gets `subgoal_dones` (N, 2) = [task_done, cost] post-action labels,
    recomputed from the stored states / observations with the env's geometry
    and cached under ~/.vadar/cache. With the 'parity' (aggSAC) reward the
    rewards are relabelled to that reward.
    """
    base_name, obstacle_cfg = resolve_obstacle_cfg(env_name, obstacle_cfg, subgoal_cfg)
    if bc_dataset_path:
        data = load_bc_dataset(bc_dataset_path)
    else:
        data = _get_upstream_dataset(base_name)
    if obstacle_cfg is None or (subgoal_cfg is None and obstacle_cfg['dense_reward_mode'] == 'none'):
        return data

    terminals = np.asarray(data['terminals'], dtype=np.float32)
    if bc_dataset_path:
        def compute():
            agent_xy, block_pose = _poses_from_observations(data['next_observations'])
            labels = _pusht_label_poses(agent_xy, block_pose, obstacle_cfg)
            masks = np.asarray(data['masks'] if 'masks' in data else 1.0 - terminals, dtype=np.float32)
            mismatch = int(((labels[:, 0] > 0.5) != (masks < 0.5)).sum())
            if mismatch:
                print(f"[pusht] note: {mismatch}/{len(labels)} relabelled task flags differ from the stored "
                      "terminations (expected only at the coverage threshold, or for datasets with other "
                      "termination rules)")
            return _cumulative_task_done(labels, terminals)
        labels = _cached_labels(_file_key(os.path.expanduser(bc_dataset_path)), obstacle_cfg, compute)
    else:
        labels = _upstream_subgoal_labels(base_name, terminals, obstacle_cfg)

    data = dict(data)
    if subgoal_cfg is not None:
        data['subgoal_dones'] = labels.astype(np.float32)
    mode = obstacle_cfg['dense_reward_mode']
    if mode == 'parity':
        task = np.where(labels[:, 0] > 0.5, float(obstacle_cfg['dense_task_bonus']),
                        float(obstacle_cfg['dense_sparse_step_cost']))
        safety = -float(obstacle_cfg['dense_safety_scale']) * labels[:, 1]
        data['rewards'] = (task + float(obstacle_cfg['dense_safety_coeff']) * safety).astype(np.float32)
    elif mode == 'pbrs':
        print("[pusht] dense_reward_mode='pbrs': the offline rewards are left unchanged.")
    return data
