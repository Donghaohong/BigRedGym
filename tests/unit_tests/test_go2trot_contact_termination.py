"""Go2 trot terminates on head/thigh contact, including self-collision.

Poses are written through the task's public state and committed with the
backend's reset, which refreshes contact forces without stepping physics.
"""

import math

import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.task_registry import select_backend, task_registry

STAND = 4 * [0.0, 0.96, -1.36]  # canonical FL, FR, RL, RR; hip, thigh, calf
FOLDED = 4 * [0.0, 1.5, -2.7]


@pytest.fixture
def task():
    cfg = Go2TrotCfg()
    cfg.seed = 0
    cfg.env.num_envs = 1
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    task_registry.convert_frequencies_to_params(cfg, Go2TrotRunnerCfg())
    backend = select_backend(cfg, "cpu", "mujoco")
    try:
        yield Go2Trot(cfg, "cpu", True, backend)
    finally:
        backend.close()


def _place(task, z, joint_angles, pitch_deg=0.0):
    """Commit a static pose and return the termination flag it produces."""
    half = math.radians(pitch_deg) / 2
    task.root_states[0, :] = 0.0
    task.root_states[0, 2] = z
    task.root_states[0, 3:7] = torch.tensor([0.0, math.sin(half), 0.0, math.cos(half)])
    task.dof_pos[0, :] = 0.0
    task.dof_pos[0, task.actuated_dof_indices] = torch.tensor(joint_angles)
    task.dof_vel[0, :] = 0.0
    task._backend.reset_state(torch.tensor([True]))
    task.terminated[:] = False
    task._check_terminations_and_timeouts()
    return bool(task.terminated[0])


def _touching(task):
    force = task.contact_forces[0].norm(dim=-1)
    return {task._backend.body_names[i] for i in torch.nonzero(force > 1.0).flatten()}


def test_termination_bodies_are_base_head_and_thighs(task):
    names = {task._backend.body_names[i] for i in task.termination_contact_indices}
    legs = ("FL", "FR", "RL", "RR")
    expected = {"base", "Head_upper", "Head_lower"}
    expected |= {f"{leg}_thigh" for leg in legs}
    expected |= {f"{leg}_thigh_rotor" for leg in legs}
    assert names == expected


def test_standing_on_feet_does_not_terminate(task):
    terminated = _place(task, 0.30, STAND)
    touching = _touching(task)
    assert touching and all(name.endswith("_foot") for name in touching)
    assert not terminated


def test_head_on_ground_terminates(task):
    terminated = _place(task, 0.25, FOLDED, pitch_deg=50.0)
    assert "Head_lower" in _touching(task)
    assert terminated


def test_front_thigh_self_collision_terminates_in_the_air(task):
    joints = list(STAND)
    joints[0], joints[3] = -1.0, 1.0  # FL / FR hips swung inward
    terminated = _place(task, 1.5, joints)
    touching = _touching(task)
    assert {"FL_thigh", "FR_thigh"} <= touching
    assert not any(name.endswith("_foot") for name in touching)  # no ground
    assert terminated


def test_calf_only_self_collision_does_not_terminate(task):
    joints = list(STAND)
    joints[0], joints[3] = -0.8, 0.8
    terminated = _place(task, 1.5, joints)
    assert _touching(task) == {"FL_calf", "FR_calf"}
    assert not terminated
