"""Quantitative base-height tracking test for a trained go2trot policy.

Holds the velocity command at zero and steps the base-height command through
a fixed schedule (settle at mid height, then low -> high -> low -> high),
recording commanded and actual height for every environment.

* The environment and runner configs are the ones saved with the run
  (``original_cfg``), so the evaluated observation layout, command range and
  scales are the model's own. Only evaluation overrides are applied on top:
  domain randomization and pushes off (before the env is built), command
  resampling off, long episode.
* The RNG is seeded right before the single reset; nothing is reset during
  the test, so a failure is never stitched into a successful-looking curve.
* Failures use one fixed criterion for every model, independent of the
  model's own termination settings: net contact force above 1 N on any body
  whose name contains one of ``--forbidden_bodies`` (self-collision counts),
  or full tilt angle above ``--tilt_limit_deg``. For every environment the
  first failure is recorded separately in each phase -- "init" (from the
  reset through the settle segment) and "tracking" (the command steps) -- so
  an early fall does not hide later tracking failures. Every exported file
  carries the status and these events.

Metrics per command segment, over all environments:
  * steady-state error: mean and RMS of (height - command) over the last
    ``--steady`` seconds of the segment;
  * settling time: first time after the step from which |error| stays within
    ``--band`` for at least ``--hold`` seconds (NaN if never).

Usage:
    uv run --frozen scripts/plot_height_tracking.py \
        --experiment_name go2trot_height --load_run <run> --checkpoint 550
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import gym.envs  # noqa: E402, F401
from gym import GYM_ROOT_DIR  # noqa: E402
from gym.utils.helpers import set_seed  # noqa: E402
from gym.utils.task_registry import task_registry  # noqa: E402

CONTACT_THRESHOLD_N = 1.0


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="evaluation seed")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--low", type=float, default=None, help="default: range min")
    parser.add_argument("--high", type=float, default=None, help="default: range max")
    parser.add_argument("--settle", type=float, default=3.0, help="s at mid height")
    parser.add_argument("--segment", type=float, default=6.0, help="s per step")
    parser.add_argument("--steady", type=float, default=2.0)
    parser.add_argument("--band", type=float, default=0.02, help="m")
    parser.add_argument("--hold", type=float, default=0.5, help="s")
    parser.add_argument("--tilt_limit_deg", type=float, default=30.0)
    parser.add_argument(
        "--forbidden_bodies", nargs="+", default=["base", "Head", "thigh"]
    )
    return parser.parse_args(argv)


def setup(args):
    env_cfg, train_cfg = task_registry.get_cfgs(
        "go2trot",
        original_cfg=True,
        experiment_name=args.experiment_name,
        load_run=args.load_run,
    )
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 1000
    env_cfg.commands.resampling_time = 9999
    env_cfg.commands.resample_base_height = False
    env_cfg.push_robots.toggle = False
    env_cfg.domain_randomization.startup.contact_friction_range = None
    env_cfg.domain_randomization.startup.link_mass_scale_range = None
    env_cfg.domain_randomization.episode.scale_ranges = {}
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.seed = train_cfg.seed = args.seed

    train_cfg.runner.device = "cpu"
    train_cfg.runner.resume = True
    train_cfg.runner.checkpoint = args.checkpoint
    train_cfg.logging.enable_local_saving = False

    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    task_registry.set_log_dir_name(train_cfg)
    set_seed(args.seed)
    env = task_registry.make_env(
        "go2trot", env_cfg, device="cpu", headless=True, backend="mujoco"
    )
    runner = task_registry.make_alg_runner(env, train_cfg)
    runner.switch_to_eval()
    return env, runner, train_cfg._original_cfg_source_dir


def schedule(settle, segment, low, high):
    """[(start_s, end_s, command_m)], starting with a settle phase at mid."""
    segments, t = [(0.0, settle, 0.5 * (low + high))], settle
    for h in (low, high, low, high):
        segments.append((t, t + segment, h))
        t += segment
    return segments


def settling_time(t, err, step_time, band, hold, dt):
    """Seconds from the step until |err| first stays within band for hold."""
    inside = err.abs() <= band
    need = max(1, round(hold / dt))
    for i in range(len(t) - need + 1):
        if inside[i : i + need].all():
            return float(t[i]) - step_time
    return math.nan


def first_failures(contact_fail, tilt_fail, t, settle_end, body_names):
    """First failure of each env in each phase: [{env, init, tracking}].

    ``contact_fail`` is [T, N, B] booleans over the forbidden bodies,
    ``tilt_fail`` is [T, N]; ``t`` holds each step's end time. A phase entry
    is None or {time_s, reason}; "init" covers t <= settle_end.
    """
    any_fail = contact_fail.any(dim=2) | tilt_fail
    phases = {"init": t <= settle_end, "tracking": t > settle_end}
    failures = []
    for env in range(any_fail.shape[1]):
        record = {"env": env}
        for phase, window in phases.items():
            steps = torch.nonzero(any_fail[:, env] & window).flatten()
            if len(steps) == 0:
                record[phase] = None
                continue
            k = int(steps[0])
            reasons = [
                f"contact:{body_names[b]}"
                for b in torch.nonzero(contact_fail[k, env]).flatten().tolist()
            ]
            if tilt_fail[k, env]:
                reasons.append("tilt")
            record[phase] = {
                "time_s": round(float(t[k]), 4),
                "reason": "+".join(reasons),
            }
        failures.append(record)
    return failures


def run(env, runner, args, segments):
    set_seed(args.seed)
    env.timed_out[:] = True
    env.reset()
    env.commands[:] = 0.0
    forbidden = env._backend.build_contact_indices(args.forbidden_bodies, "cpu")
    tilt_limit = math.radians(args.tilt_limit_deg)
    steps = round(segments[-1][1] / env.dt)
    t, cmd, height, tilt, contact_fail, tilt_fail = [], [], [], [], [], []
    for k in range(steps):
        now = k * env.dt
        command = next(h for start, end, h in segments if start <= now < end)
        env.base_height_command[:] = command
        runner.set_actions(
            runner.actor_cfg["actions"], runner.get_inference_actions(), False
        )
        env.step()
        g = env.projected_gravity
        angle = torch.atan2(g[:, :2].norm(dim=1), -g[:, 2])
        force = env.contact_forces[:, forbidden].norm(dim=-1)
        t.append(now + env.dt)
        cmd.append(command)
        height.append(env.base_height[:, 0].clone())
        tilt.append(angle.clone())
        contact_fail.append(force > CONTACT_THRESHOLD_N)
        tilt_fail.append(angle > tilt_limit)
    names = [env._backend.body_names[i] for i in forbidden.tolist()]
    t = torch.tensor(t)
    failures = first_failures(
        torch.stack(contact_fail), torch.stack(tilt_fail), t, segments[0][1], names
    )
    return t, torch.tensor(cmd), torch.stack(height, 1), torch.stack(tilt, 1), failures


def segment_metrics(t, height, segments, failures, steady_s, band, hold):
    dt = float(t[1] - t[0])
    rows = []
    for index, (start, end, h) in enumerate(segments[1:], start=1):
        sel = (t > start) & (t <= end)
        steady = sel & (t > end - steady_s)
        err = height[:, sel] - h
        steady_err = height[:, steady] - h
        settle = [
            settling_time(t[sel], err[e], start, band, hold, dt)
            for e in range(height.shape[0])
        ]
        rows.append(
            {
                "segment": index,
                "command_m": h,
                "mean_err_m": steady_err.mean().item(),
                "rms_err_m": steady_err.square().mean().sqrt().item(),
                "worst_env_rms_m": steady_err.square().mean(dim=1).sqrt().max().item(),
                "settle_s_median": torch.tensor(settle).nanmedian().item(),
                # NaN as soon as one env never settles: worst is unbounded.
                "settle_s_worst": math.nan
                if any(math.isnan(x) for x in settle)
                else max(settle),
                "envs_not_settled": sum(math.isnan(s) for s in settle),
                "envs_first_tracking_failure_here": sum(
                    f["tracking"] is not None and start < f["tracking"]["time_s"] <= end
                    for f in failures
                ),
            }
        )
    return rows


def summarize(failures):
    failed = [f for f in failures if f["init"] or f["tracking"]]
    return {
        "status": "FAILED" if failed else "PASS",
        "envs": len(failures),
        "envs_failed": len(failed),
        "envs_failed_init": sum(f["init"] is not None for f in failures),
        "envs_failed_tracking": sum(f["tracking"] is not None for f in failures),
        "first_failures": failed,
    }


def plot(path, t, cmd, height, tilt, segments, rows, summary, args):
    fig, (ax_h, ax_t) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1]
    )
    for e in range(height.shape[0]):
        ax_h.plot(t, height[e], color="tab:blue", alpha=0.25, lw=0.8)
    ax_h.plot(t, height.mean(dim=0), color="tab:blue", lw=1.8, label="actual (mean)")
    ax_h.plot(t, cmd, color="black", ls="--", lw=1.5, label="command")
    ax_h.fill_between(
        t,
        cmd - args.band,
        cmd + args.band,
        color="gray",
        alpha=0.15,
        label=f"±{args.band * 100:.0f} cm band",
    )
    for f in summary["first_failures"]:
        for phase, color in (("init", "tab:orange"), ("tracking", "tab:red")):
            if f[phase] is not None:
                for ax in (ax_h, ax_t):
                    ax.axvline(f[phase]["time_s"], color=color, lw=0.8, alpha=0.6)
    ax_h.set_ylabel("base height [m]")
    ax_h.legend(loc="lower right")
    status = (
        f"FAILED: {summary['envs_failed']}/{summary['envs']} envs "
        f"(init {summary['envs_failed_init']}, "
        f"tracking {summary['envs_failed_tracking']})"
        if summary["envs_failed"]
        else "PASS: no failures"
    )
    ax_h.set_title(
        f"{args.experiment_name}/{args.load_run} model_{args.checkpoint}, "
        f"eval seed {args.seed}, {height.shape[0]} envs - {status}",
        fontsize=10,
    )
    top = ax_h.get_ylim()[1]
    for row, (start, end, _) in zip(rows, segments[1:]):
        label = f"RMS {row['rms_err_m'] * 100:.1f} cm"
        ax_h.text(0.5 * (start + end), top, label, ha="center", va="top", fontsize=8)
    ax_t.plot(t, tilt.max(dim=0).values.rad2deg(), color="tab:red", lw=1)
    ax_t.axhline(args.tilt_limit_deg, color="tab:red", ls=":", lw=1)
    ax_t.set_ylabel("max tilt [deg]")
    ax_t.set_xlabel("time [s]  (first failure per env: orange = init, red = tracking)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    args = get_args()
    with torch.no_grad():
        env, runner, cfg_source = setup(args)
        low, high = env.cfg.commands.ranges.base_height
        low = low if args.low is None else args.low
        high = high if args.high is None else args.high
        segments = schedule(args.settle, args.segment, low, high)
        t, cmd, height, tilt, failures = run(env, runner, args, segments)
    rows = segment_metrics(
        t, height, segments, failures, args.steady, args.band, args.hold
    )
    summary = summarize(failures)

    # Next to the checkpoint, under the gitignored logs/ tree.
    run_dir = Path(GYM_ROOT_DIR) / "logs" / args.experiment_name / args.load_run
    stem = f"height_tracking_model{args.checkpoint}_seed{args.seed}"
    png, table, record = (run_dir / f"{stem}.{x}" for x in ("png", "csv", "json"))
    plot(png, t, cmd, height, tilt, segments, rows, summary, args)
    with open(table, "w", newline="") as f:
        fields = ["status", *rows[0]]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"status": summary["status"], **r} for r in rows)
    record.write_text(
        json.dumps(
            {
                "args": vars(args),
                "config_source": cfg_source,
                "contact_threshold_n": CONTACT_THRESHOLD_N,
                "segments": segments,
                **summary,
                "segment_metrics": rows,
            },
            indent=2,
        )
    )

    print(
        f"evaluation: {summary['status']} - {summary['envs_failed']}/"
        f"{summary['envs']} envs failed (init {summary['envs_failed_init']}, "
        f"tracking {summary['envs_failed_tracking']})"
    )
    for f in summary["first_failures"]:
        events = [
            f"{phase}: {f[phase]['reason']} at {f[phase]['time_s']:.2f}s"
            for phase in ("init", "tracking")
            if f[phase] is not None
        ]
        print(f"  env {f['env']}: " + "; ".join(events))
    for r in rows:
        median, worst = (
            "not settled" if math.isnan(x) else f"{x:.2f}s"
            for x in (r["settle_s_median"], r["settle_s_worst"])
        )
        print(
            f"segment {r['segment']} cmd {r['command_m']:.3f} m: "
            f"mean err {r['mean_err_m'] * 100:+.1f} cm, "
            f"RMS {r['rms_err_m'] * 100:.1f} cm "
            f"(worst env {r['worst_env_rms_m'] * 100:.1f} cm), "
            f"settle median {median} / worst {worst}, "
            f"{r['envs_not_settled']} not settled, "
            f"{r['envs_first_tracking_failure_here']} first tracking failure(s) here"
        )
    print(f"wrote {png}\nwrote {table}\nwrote {record}")


if __name__ == "__main__":
    main()
