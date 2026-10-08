"""Go2 trot base-height command: buffer, resampling, reward, and observation."""

import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.interfaces.teleop_bindings import TeleopCommands
from gym.utils.task_registry import select_backend, task_registry


@pytest.fixture
def task():
    cfg = Go2TrotCfg()
    cfg.seed = 7
    cfg.env.num_envs = 2
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    cfg.init_state.reset_mode = "reset_to_basic"
    task_registry.convert_frequencies_to_params(cfg, Go2TrotRunnerCfg())
    backend = select_backend(cfg, "cpu", "mujoco")
    try:
        yield Go2Trot(cfg, "cpu", True, backend)
    finally:
        backend.close()


def _in_range(task, heights):
    low, high = task.cfg.commands.ranges.base_height
    return bool(torch.all((heights >= low) & (heights <= high)))


def test_height_command_buffer_is_sampled_within_range(task):
    command = task.base_height_command
    assert command.shape == (2, 1)
    assert command.dtype == torch.float and command.device == torch.device("cpu")
    assert _in_range(task, command)


def test_resampling_only_updates_masked_envs_in_place(task):
    buffer = task.base_height_command
    buffer.fill_(1.0)  # sentinel outside the command range
    task._resample_commands(torch.tensor([True, False]))
    assert task.base_height_command is buffer
    assert _in_range(task, buffer[0])
    assert buffer[1].item() == 1.0


def test_sparse_reset_resamples_only_the_reset_env(task):
    task.base_height_command.fill_(1.0)
    task._reset_idx(torch.tensor([False, True]))
    assert task.base_height_command[0].item() == 1.0
    assert _in_range(task, task.base_height_command[1])


def _step_across_resampling_boundary(task):
    period = int(task.cfg.commands.resampling_time / task.dt)
    task.episode_length_buf[:] = period - 1
    task.step()  # _post_decimation_step advances to the boundary and resamples


def test_training_resamples_height_at_the_resampling_boundary(task):
    task.base_height_command.fill_(1.0)  # sentinel outside the command range
    _step_across_resampling_boundary(task)
    assert _in_range(task, task.base_height_command)


def test_teleop_height_survives_resampling_boundary_and_reset(task):
    commands = TeleopCommands(task)
    for _ in range(20):
        commands.apply("down")
    low = task.cfg.commands.ranges.base_height[0]
    expected = torch.full((2, 1), low)
    seeded_velocity = task.commands.clone()
    _step_across_resampling_boundary(task)
    # The boundary really fired: velocity commands (upstream behavior) moved.
    assert not torch.equal(task.commands, seeded_velocity)
    torch.testing.assert_close(task.base_height_command, expected)
    commands.apply("reset")
    torch.testing.assert_close(task.base_height_command, expected)


def test_tracking_reward_peaks_at_command_and_decreases_with_error(task):
    height = task.base_height.clone()
    rewards = []
    for offset in (0.0, 0.05, 0.10, -0.10):
        task.base_height_command[:] = height + offset
        rewards.append(task._reward_tracking_base_height())
    torch.testing.assert_close(rewards[0], torch.ones(2))
    assert torch.all(rewards[1] < rewards[0]) and torch.all(rewards[2] < rewards[1])
    torch.testing.assert_close(rewards[3], rewards[2])
    scale = task.cfg.reward_settings.base_height_tracking_scale
    sigma = task.cfg.reward_settings.tracking_sigma
    expected = torch.exp(-torch.tensor((0.05 / scale) ** 2 / sigma))
    torch.testing.assert_close(rewards[1], expected.expand(2))


def test_tracking_bandwidth_is_independent_of_observation_scale(task):
    """Tuning the height-reward bandwidth must not touch the obs scaling."""
    task.base_height_command[:] = task.base_height + 0.05
    obs_before = task.get_state("base_height_command").clone()
    sigma = task.cfg.reward_settings.tracking_sigma
    for scale in (0.3, 0.1):
        task.cfg.reward_settings.base_height_tracking_scale = scale
        expected = torch.exp(-torch.tensor((0.05 / scale) ** 2 / sigma))
        torch.testing.assert_close(
            task._reward_tracking_base_height(), expected.expand(2)
        )
    torch.testing.assert_close(task.get_state("base_height_command"), obs_before)
    assert task.scales["base_height"] == task.cfg.scaling.base_height


def test_height_command_observation_is_scaled(task):
    task.base_height_command[:] = torch.tensor([[0.15], [0.30]])
    torch.testing.assert_close(
        task.get_state("base_height_command"),
        task.base_height_command / task.cfg.scaling.base_height_command,
    )


def test_runner_uses_height_command_and_drops_fixed_height_floor():
    runner = Go2TrotRunnerCfg()
    assert "base_height_command" in runner.actor.obs
    assert "base_height_command" in runner.critic.obs
    assert runner.critic.reward.weights.tracking_base_height > 0
    assert runner.critic.reward.weights.min_base_height == 0
