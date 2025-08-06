# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg , RigidObject , RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform , quat_mul, quat_from_euler_xyz
from isaaclab_assets import franka

from torch.utils.tensorboard import SummaryWriter
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema


@configclass
class FrankaReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 21
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path= f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            # usd_path= f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
            # usd_path= f"{ISAAC_NUCLEUS_DIR}/Robots/Kinova/Gen3/gen3n7_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                # "panda_joint1": -1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                # "panda_joint4": -2.239,
                "panda_joint4": -1.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # # cabinet
    # cabinet = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/Cabinet",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0, 0.4),
    #         rot=(0.1, 0.0, 0.0, 0.0),
    #         joint_pos={
    #             "door_left_joint": 0.0,
    #             "door_right_joint": 0.0,
    #             "drawer_bottom_joint": 0.0,
    #             "drawer_top_joint": 0.0,
    #         },
    #     ),
    #     actuators={
    #         "drawers": ImplicitActuatorCfg(
    #             joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=10.0,
    #             damping=1.0,
    #         ),
    #         "doors": ImplicitActuatorCfg(
    #             joint_names_expr=["door_left_joint", "door_right_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=10.0,
    #             damping=2.5,
    #         ),
    #     },
    # )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )


    # cuboid
    cuboid = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cuboid",
        # spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",

        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.1, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            # collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, 
            
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )



    # cuboid
    goal = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, 

            )
        ),
        # spawn=sim_utils.MeshCuboidCfg(
        #     size=(0.1, 0.1, 0.1),
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        #     # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #     # collision_props=sim_utils.CollisionPropertiesCfg(),
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=True, 
            
        #     )
        # ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


    door = RigidObjectCfg(
            prim_path="/World/envs/env_.*/door",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/root/Documents/door_green6.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, 

                )
            ),
            
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )


    muffler = RigidObjectCfg(
            prim_path="/World/envs/env_.*/muffler",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"/root/Documents/muffler2.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),

            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
        )


    action_scale = 7.5
    dof_velocity_scale = 0.1
    # reward scales
    dist_reward_scale = 1.5
    # rot_reward_scale = 1.5
    rot_reward_scale = 1.0
    # open_reward_scale = 20.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0
    collision_penalty_scale: float = 1.5   


class FrankaReachEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.tb_writer = SummaryWriter(log_dir=f"/isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/franka_reach/logs/{self.__class__.__name__}")
        self._tb_step = 0
        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        # self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        # self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            # (self.num_envs, 1)
        # )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs, 1)
        # )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.cube_collisions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # self._goal   = RigidObject(self.cfg.goal)
        self._cuboid = RigidObject(self.cfg.cuboid)
        self._door   = RigidObject(self.cfg.door)
        self._goal  = RigidObject(self.cfg.muffler) #muffler as goal 

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["goal"] = self._goal
        self.scene.articulations["cuboid"] = self._cuboid
        self.scene.articulations["door"] = self._door
        # self.scene.articulations["muffler"] = self._muffler

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # dist = torch.norm(self.robot_grasp_pos - self.goal_pos, p=2, dim=-1)
        # terminated = (dist < 0.1) | self.cube_collisions
        # terminated = self.cube_collisions
        # truncated = self.episode_length_buf >= self.max_episode_length - 1

        self._compute_intermediate_values()
        dist = torch.norm(self.robot_grasp_pos - self.goal_pos, p=2, dim=-1)
        success = dist < 0.1
        # terminate on success or on any cuboid collision
        terminated = success | self.cube_collisions

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self.robot_grasp_pos,
            self.goal_pos,
            self.robot_grasp_rot,
            self.gripper_forward_axis,
            self.gripper_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.action_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # Reset cube collisions
        
        self.cube_collisions[env_ids] = False
        # cube_pos = torch.tensor([0.5, 0.0, 0.5], device=self.device).repeat(len(env_ids), 1)
        # cube_pos[:, 0] += sample_uniform(-0.2, 0.2, (len(env_ids),), self.device)
        # cube_pos[:, 1] += sample_uniform(-0.2, 0.2, (len(env_ids),), self.device)
        # self._cube.set_world_poses(cube_pos, None, env_ids=env_ids)
        root_states = self._cuboid.data.default_root_state[env_ids].clone()
        # current_gravity_status = self._cuboid.root_physx_view.get_disable_gravities()
        # current_gravity_status[env_ids] = 0  # need to check if 0 means enable or disable
        # self._cuboid.root_physx_view.set_disable_gravities(current_gravity_status, env_ids)
        
        goal_pose_range = {
            "x": (0.2, 0.2),   # Range for x position
            "y": (-0.6, -0.1), # Range for y position
            "z": (0.40, 0.80),    # Fixed z position (can be adjusted as needed)
            "roll": (-0.5, 0.5),  # Roll range
            "pitch": (0.0, 0.0), # Pitch range
            "yaw": (0.0, 0.0)    # Yaw range
        }
        
        goal_velocity_range = {
            "x": (0.0, 0.0),   # Example range for x velocity
            "y": (0.0, 0.0),   # Example range for y velocity
            "z": (0.0, 0.0),   # Example range for z velocity
            "roll": (0.0, 0.0), # Example roll velocity range
            "pitch": (0.0, 0.0),# Example pitch velocity range
            "yaw": (0.0, 0.0)   # Example yaw velocity range
        }


        range_list = [goal_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

        range_list = [goal_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        velocities = root_states[:, 7:13] + rand_samples
        # set into the physics simulation
        self._goal.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self._goal.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        self.goal_pos[env_ids] = positions     #Super important
        
        pose_range = {
            "x": (-0.2, 0.2),   # Range for x position
            "y": (0.0, 0.0), # Range for y position
            "z": (0.80, 0.80),    # Fixed z position (can be adjusted as needed)
            "roll": (-0.5, 0.5),  # Roll range
            "pitch": (0.0, 0.0), # Pitch range
            "yaw": (0.0, 0.0)    # Yaw range
        }
        
        velocity_range = {
            "x": (0.0, 0.0),   # Example range for x velocity
            "y": (0.0, 0.0),   # Example range for y velocity
            "z": (0.0, 0.0),   # Example range for z velocity
            "roll": (0.0, 0.0), # Example roll velocity range
            "pitch": (0.0, 0.0),# Example pitch velocity range
            "yaw": (0.0, 0.0)   # Example yaw velocity range
        }


        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        velocities = root_states[:, 7:13] + rand_samples
        # set into the physics simulation
        self._cuboid.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self._cuboid.write_root_velocity_to_sim(velocities, env_ids=env_ids)


        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        to_target = self.goal_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_target,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]

        self.robot_grasp_rot[env_ids], self.robot_grasp_pos[env_ids] = tf_combine(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

        cuboid_pos = self._cuboid.data.root_pos_w[env_ids]                       # (N,3)
        half_extents = torch.tensor(self.cfg.cuboid.spawn.size, device=self.device) * 0.5  # (3,)

        # world positions of ALL robot bodies: (N, num_bodies, 3)
        robot_body_pos = self._robot.data.body_pos_w[env_ids]  

        # compute per-body whether inside the cuboid box
        # Compute the difference between robot body positions and cuboid center
        # Then check if each body is within the cuboid's half extents (i.e., inside the cuboid volume)
        # If robot_body_pos or cuboid links have thickness, you may want to expand the half_extents by the robot link bounding radius
        # For simplicity, assume robot_body_pos represents the center of each link and cuboid is axis-aligned

        # Optionally, if you have per-link bounding radii/thickness, add here:
        # Example: robot_link_radii = torch.full((robot_body_pos.shape[1],), 0.025, device=self.device)
        # expanded_half_extents = half_extents + robot_link_radii.unsqueeze(0).unsqueeze(-1)
        # For now, just use half_extents

        diff = torch.abs(robot_body_pos - cuboid_pos.unsqueeze(1))  # (N, num_bodies, 3)
        self.inside = torch.all(diff <= half_extents, dim=-1)       
        
        # collision if ANY body link is inside
        self.cube_collisions[env_ids] = torch.any(self.inside, dim=1)  

        # # Visual feedback for collisions
        # if torch.any(self.cube_collisions):
        #     colliding_envs = torch.where(self.cube_collisions)[0]
        #     for env_id in colliding_envs:
        #         # Change cube color to red when colliding
        #         self.scene.set_prim_color(
        #             f"/World/envs/env_{env_id}/Cube",
        #             torch.tensor([1.0, 0.0, 0.0], device=self.device)
        #         )
    def _compute_rewards(
        self,
        actions,
        franka_grasp_pos,
        goal_pos,
        franka_grasp_rot,
        gripper_forward_axis,
        gripper_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        action_penalty_scale,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - goal_pos, p=2, dim=-1)
        dist_reward = -d**2

        # orientation alignment reward: gripper forward vs world forward
        axis = tf_vector(franka_grasp_rot, gripper_forward_axis)
        world_forward = torch.tensor([0, 0, 1], device=axis.device).repeat(num_envs, 1)
        rot_reward = torch.sum(axis * world_forward, dim=-1)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        # open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle

        collision_penalty = torch.where(
                   self.cube_collisions,
                   -self.cfg.collision_penalty_scale,
                   torch.tensor(0.0, device=self.device)
               )

        rewards = (
            dist_reward_scale * dist_reward
          + rot_reward_scale  * rot_reward
          - action_penalty_scale * action_penalty
          + collision_penalty

        )
        success = d < 0.1
        rewards = torch.where(success, rewards + 100.0, rewards)

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            # "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            # "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            # "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            # "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
            "collision_penalty" : collision_penalty.mean(),
        }

        # bonus for opening drawer properly
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)
        # ----- TensorBoard logging -----
        # compute current success rate
        dist = torch.norm(self.robot_grasp_pos - self.goal_pos, p=2, dim=-1)
        success_rate = (dist < 0.1).float().mean().item()
        # log all extras
        for name, val in self.extras["log"].items():
            # val may be a torch scalar or Python float
            scalar = val.item() if isinstance(val, torch.Tensor) else float(val)
            self.tb_writer.add_scalar(f"env/{name}", scalar, self._tb_step)
        # log success rate
        self.tb_writer.add_scalar("env/success_rate", success_rate, self._tb_step)
        self._tb_step += 1
        self.tb_writer.flush()
        # ------------------------------
        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos