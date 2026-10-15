"""Pure helpers of scripts/plot_height_tracking.py (no simulator)."""

import math

import torch

from scripts.plot_height_tracking import (
    first_failures,
    schedule,
    segment_metrics,
    settling_time,
    summarize,
    tracking_summary,
)

DT = 0.01


def test_schedule_settles_at_mid_then_low_high_low_high():
    segments = schedule(settle=3.0, segment=6.0, low=0.15, high=0.35)
    assert [round(h, 3) for _, _, h in segments] == [0.25, 0.15, 0.35, 0.15, 0.35]
    assert [(s, e) for s, e, _ in segments] == [
        (0.0, 3.0),
        (3.0, 9.0),
        (9.0, 15.0),
        (15.0, 21.0),
        (21.0, 27.0),
    ]


def test_settling_time_needs_the_full_hold_inside_the_band():
    t = torch.arange(1, 201) * DT  # 2 s after a step at t=0
    err = torch.full((200,), 0.10)
    err[50:60] = 0.0  # 0.1 s inside: too short
    err[100:] = 0.01  # inside from t=1.01 s onwards
    assert math.isclose(settling_time(t, err, 0.0, 0.02, 0.5, DT), 1.01, abs_tol=1e-6)
    assert math.isnan(settling_time(t, torch.full((200,), 0.1), 0.0, 0.02, 0.5, DT))


def _events(steps=100, envs=3, bodies=2):
    return (
        torch.zeros(steps, envs, bodies, dtype=torch.bool),
        torch.zeros(steps, envs, dtype=torch.bool),
        torch.arange(1, steps + 1) * DT,
    )


def test_first_failures_are_kept_separately_per_phase():
    contact, tilt, t = _events()
    contact[4, 0, 1] = True  # env 0: init-phase contact at t=0.05
    tilt[69, 0] = True  # env 0: tracking-phase tilt at t=0.70
    tilt[79, 1] = True  # env 1: tracking only
    failures = first_failures(contact, tilt, t, settle_end=0.5, body_names=["a", "b"])
    assert failures[0]["init"] == {"time_s": 0.05, "reason": "contact:b"}
    assert failures[0]["tracking"] == {"time_s": 0.7, "reason": "tilt"}
    assert failures[1]["init"] is None and failures[1]["tracking"]["time_s"] == 0.8
    assert failures[2] == {"env": 2, "init": None, "tracking": None}

    summary = summarize(failures)
    assert summary["stability_status"] == "FAILED"
    assert (summary["envs_failed"], summary["envs_failed_init"]) == (2, 1)
    assert summary["envs_failed_tracking"] == 2
    assert [f["env"] for f in summary["first_failures"]] == [0, 1]


def test_simultaneous_reasons_are_all_reported():
    contact, tilt, t = _events()
    contact[59, 0, :] = True
    tilt[59, 0] = True
    failures = first_failures(contact, tilt, t, 0.5, ["Head_lower", "FL_thigh"])
    assert failures[0]["tracking"]["reason"] == (
        "contact:Head_lower+contact:FL_thigh+tilt"
    )


def test_no_failures_is_stable():
    contact, tilt, t = _events()
    summary = summarize(first_failures(contact, tilt, t, 0.5, ["a", "b"]))
    assert summary["stability_status"] == "STABLE"


def test_segment_metrics_report_errors_settling_and_failures():
    segments = schedule(settle=1.0, segment=2.0, low=0.15, high=0.35)
    t = torch.arange(1, round(segments[-1][1] / DT) + 1) * DT
    command = torch.tensor([next(h for s, e, h in segments if s < x <= e) for x in t])
    height = torch.stack([command, command - 0.05])  # env 1 sits 5 cm low
    failures = [
        {"env": 0, "init": None, "tracking": None},
        {"env": 1, "init": None, "tracking": {"time_s": 3.5, "reason": "tilt"}},
    ]
    rows = segment_metrics(t, height, segments, failures, 1.0, 0.02, 0.5)
    assert [r["segment"] for r in rows] == [1, 2, 3, 4]
    first = rows[0]
    assert math.isclose(first["mean_err_m"], -0.025, abs_tol=1e-6)
    assert math.isclose(first["worst_env_rms_m"], 0.05, abs_tol=1e-6)
    assert first["envs_not_settled"] == 1 and math.isnan(first["settle_s_worst"])
    assert [r["envs_first_tracking_failure_here"] for r in rows] == [0, 1, 0, 0]
    assert first["settle_s_per_env"][1] is None
    assert [r["envs_steady_mean_in_band"] for r in rows] == [1, 1, 1, 1]


def test_tracking_is_reported_apart_from_stability():
    """A stable policy that ignores the command must not look like tracking."""
    segments = schedule(settle=1.0, segment=3.0, low=0.15, high=0.35)
    t = torch.arange(1, round(segments[-1][1] / DT) + 1) * DT
    command = torch.tensor([next(h for s, e, h in segments if s < x <= e) for x in t])
    late = command.clone()  # env 1 reaches each command 1.5 s after the step
    for start, _, h in segments[1:]:
        late[(t > start) & (t <= start + 1.5)] = 0.25
    stuck = torch.full_like(command, 0.27)  # env 2 ignores the command
    height = torch.stack([command, late, stuck])
    failures = [{"env": e, "init": None, "tracking": None} for e in range(3)]
    rows = segment_metrics(t, height, segments, failures, 1.0, 0.02, 0.5)

    assert summarize(failures)["stability_status"] == "STABLE"
    tracking = tracking_summary(rows)
    assert set(tracking) == {"0.150", "0.350"}
    for v in tracking.values():  # 2 segments x 3 envs per height
        assert v["env_segments"] == 6
        assert (v["settled_within_1s"], v["settled_after_1s"]) == (2, 2)
        assert v["not_settled"] == 2 and v["steady_mean_in_band"] == 4
        assert math.isclose(v["settle_s_max"], 1.5, abs_tol=0.02)
