# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.length_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        geometry_scale = getattr(self.cfg, "geometry_scale", 1.0)
        slope = difficulty * 0.4
        amplitude = geometry_scale * (0.01 + 0.07 * difficulty)
        step_height = geometry_scale * (0.05 + 0.18 * difficulty)
        discrete_obstacles_height = geometry_scale * (0.05 + difficulty * 0.1)
        stepping_stones_size = geometry_scale * 1.5 * (1.05 - difficulty)
        stone_distance = geometry_scale * (0.05 if difficulty==0 else 0.1)
        gap_size = geometry_scale * 1. * difficulty
        pit_depth = geometry_scale * 1. * difficulty
        platform_size = geometry_scale * 3.0
        plane_mix = float(getattr(self.cfg, "mixed_plane_probability", 0.0))
        if plane_mix > 0.0 and choice < plane_mix:
            return terrain
        if plane_mix > 0.0:
            choice = (choice - plane_mix) / (1.0 - plane_mix)
        stair_mix = float(getattr(self.cfg, "mixed_stair_probability", 0.0))
        if stair_mix > 0.0 and choice < stair_mix:
            height_range = getattr(self.cfg, "mixed_stair_height_range", None)
            if height_range is None:
                stair_height = self.cfg.up_down_stair_height
            else:
                stair_height = height_range[0] + difficulty * (height_range[1] - height_range[0])
            up_down_stairs_terrain(
                terrain, step_width=geometry_scale * 0.30,
                step_height=stair_height, start_distance=3.5,
                num_steps=6, top_width=geometry_scale * 0.9,
                flat_width=geometry_scale * 2.5,
            )
            return terrain
        if stair_mix > 0.0:
            choice = (choice - stair_mix) / (1.0 - stair_mix)
        slope_mix = float(getattr(self.cfg, "mixed_slope_probability", 0.0))
        if slope_mix > 0.0 and choice < slope_mix:
            angle_range = getattr(self.cfg, "mixed_slope_angle_range_deg", None)
            if angle_range is None:
                angle_deg = 45.0
            else:
                # Curriculum rows use difficulty 0.0..0.9; make level 9 reach
                # the requested upper endpoint exactly.
                fraction = np.clip(difficulty / 0.9, 0.0, 1.0)
                angle_deg = angle_range[0] + fraction * (
                    angle_range[1] - angle_range[0]
                )
            start_distance = getattr(self.cfg, "slope_ramp_start_distance", None)
            if start_distance is None:
                start_distance = self.env_length / 2.0 + 1.1
            inclined_ramp_terrain(terrain, angle_deg, start_distance)
            return terrain
        if slope_mix > 0.0:
            choice = (choice - slope_mix) / (1.0 - slope_mix)
        continuous_mix = float(getattr(self.cfg, "mixed_continuous_uneven_probability", 0.0))
        if continuous_mix > 0.0 and choice < continuous_mix:
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-amplitude,
                max_height=amplitude,
                step=0.005,
                downsampled_scale=0.1,
            )
            return terrain
        if continuous_mix > 0.0:
            choice = (choice - continuous_mix) / (1.0 - continuous_mix)
        fixed_slope_angle = getattr(self.cfg, "fixed_slope_angle_deg", None)
        if fixed_slope_angle is not None:
            start_distance = getattr(self.cfg, "slope_ramp_start_distance", None)
            if start_distance is None:
                start_distance = 2.5
            inclined_ramp_terrain(terrain, fixed_slope_angle, start_distance)
        elif getattr(self.cfg, "continuous_up_stairs", False):
            height_range = getattr(self.cfg, "up_down_stair_height_range", None)
            if height_range is None:
                stair_height = self.cfg.up_down_stair_height
            else:
                stair_height = height_range[0] + difficulty * (
                    height_range[1] - height_range[0]
                )
            continuous_up_stairs_terrain(
                terrain,
                step_width=geometry_scale * 0.30,
                step_height=stair_height,
                start_distance=3.5,
            )
        elif getattr(self.cfg, "up_down_stairs", False):
            height_range = getattr(self.cfg, "up_down_stair_height_range", None)
            if height_range is None:
                stair_height = self.cfg.up_down_stair_height
            else:
                stair_height = height_range[0] + difficulty * (
                    height_range[1] - height_range[0]
                )
            up_down_stairs_terrain(
                terrain,
                step_width=geometry_scale * 0.30,
                step_height=stair_height,
                # Leave a longer flat approach and exit so oblique acceptance
                # runs do not start or finish against a tile boundary.
                start_distance=3.5,
                num_steps=6,
                top_width=geometry_scale * 0.9,
                flat_width=geometry_scale * 2.5,
            )
        elif getattr(self.cfg, "continuous_uneven", False):
            # Fill the complete tile, including the spawn region.  Difficulty
            # levels 0..4 produce approximately 1..4 cm signed height relief.
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-amplitude,
                max_height=amplitude,
                step=0.005,
                downsampled_scale=0.1,
            )
        elif choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=geometry_scale * 0.30, step_height=step_height, platform_size=platform_size)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = geometry_scale * 1.
            rectangle_max_size = geometry_scale * 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=platform_size)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        local_origin_x = getattr(self.cfg, "spawn_offset_x", None)
        if local_origin_x is None:
            local_origin_x = self.env_length / 2.
        env_origin_x = i * self.env_length + local_origin_x
        env_origin_y = (j + 0.5) * self.env_width
        x1 = max(0, int((local_origin_x - 1) / terrain.horizontal_scale))
        x2 = min(terrain.width, int((local_origin_x + 1) / terrain.horizontal_scale))
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def up_down_stairs_terrain(terrain, step_width, step_height, start_distance,
                           num_steps, top_width, flat_width):
    """Fill a tile with repeatable approach/up/platform/down/exit cycles."""
    x_m = np.arange(terrain.width) * terrain.horizontal_scale
    levels = np.zeros(terrain.width, dtype=np.int32)
    up_length = num_steps * step_width
    cycle_length = 2. * up_length + top_width + flat_width
    active = x_m >= start_distance
    phase = np.mod(x_m[active] - start_distance, cycle_length)

    active_levels = np.zeros_like(phase, dtype=np.int32)
    ascending = phase < up_length
    active_levels[ascending] = np.minimum(
        np.floor(phase[ascending] / step_width).astype(np.int32) + 1,
        num_steps,
    )
    on_top = (phase >= up_length) & (phase < up_length + top_width)
    active_levels[on_top] = num_steps
    descending = (
        (phase >= up_length + top_width)
        & (phase < 2. * up_length + top_width)
    )
    active_levels[descending] = np.maximum(
        num_steps - 1
        - np.floor(
            (phase[descending] - up_length - top_width) / step_width
        ).astype(np.int32),
        0,
    )
    levels[active] = active_levels
    raw_step = int(np.round(step_height / terrain.vertical_scale))
    terrain.height_field_raw[:, :] = (levels * raw_step)[:, None]


def continuous_up_stairs_terrain(terrain, step_width, step_height,
                                 start_distance):
    """Flat launch area followed by stairs that rise to the end of the tile."""
    x_m = np.arange(terrain.width) * terrain.horizontal_scale
    levels = np.zeros(terrain.width, dtype=np.int32)
    active = x_m >= start_distance
    levels[active] = (
        np.floor((x_m[active] - start_distance) / step_width).astype(np.int32)
        + 1
    )
    raw_step = int(np.round(step_height / terrain.vertical_scale))
    terrain.height_field_raw[:, :] = (levels * raw_step)[:, None]


def inclined_ramp_terrain(terrain, angle_deg, start_distance):
    """Fill a tile with a flat approach followed by an exact uphill ramp."""
    x_m = np.arange(terrain.width) * terrain.horizontal_scale
    rise_m = np.maximum(x_m - start_distance, 0.0) * np.tan(
        np.deg2rad(angle_deg)
    )
    raw_height = np.rint(rise_m / terrain.vertical_scale).astype(np.int16)
    terrain.height_field_raw[:, :] = raw_height[:, None]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
