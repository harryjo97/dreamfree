"""Build a self-contained HTML gallery for RoboCasa evaluation videos."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

VIDEO_FPS = 20
OUTPUT_NAME = "results.html"


def _required_sequence(results: Mapping[str, Any], key: str) -> list[Any]:
    value = results.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"RoboCasa results field {key!r} must be a sequence, got: {type(value).__name__}"
        )
    return list(value)


def _build_subgoal_page_data(
    results: Mapping[str, Any], episode_indices: Sequence[int]
) -> dict[str, Any] | None:
    diagnostics = False
    if diagnostics is None:
        return None
    if not isinstance(diagnostics, Mapping):
        raise ValueError("RoboCasa subgoal_diagnostics must be an object when present.")
    if diagnostics.get("schema_version") != 1:
        raise ValueError(
            f"RoboCasa subgoal_diagnostics.schema_version must be 1, got {diagnostics.get('schema_version')!r}."
        )
    if diagnostics.get("mode") != "ordered_ever":
        raise ValueError(
            f"RoboCasa subgoal_diagnostics.mode must be 'ordered_ever', got {diagnostics.get('mode')!r}."
        )
    episode_records = diagnostics.get("episodes")
    if isinstance(episode_records, (str, bytes)) or not isinstance(episode_records, Sequence):
        raise ValueError("RoboCasa subgoal_diagnostics.episodes must be a sequence.")
    if len(episode_records) != len(episode_indices):
        raise ValueError(
            f"RoboCasa subgoal diagnostic episode count does not match evaluation episodes: diagnostics={len(episode_records)} evaluation={len(episode_indices)}."
        )
    expected_ids: list[str] | None = None
    expected_metadata: dict[str, tuple[str, str, int]] = {}
    achieved_steps: dict[str, list[int]] = {}
    episodes_by_index: dict[int, dict[str, Any]] = {}
    for record in episode_records:
        if not isinstance(record, Mapping):
            raise ValueError("Each RoboCasa subgoal diagnostic episode must be an object.")
        episode_idx = record.get("episode_idx")
        if not isinstance(episode_idx, int) or isinstance(episode_idx, bool):
            raise ValueError(
                f"Subgoal diagnostic episode_idx must be an integer, got {episode_idx!r}."
            )
        if episode_idx in episodes_by_index:
            raise ValueError(f"Duplicate subgoal diagnostic episode_idx={episode_idx}.")
        raw_subgoals = record.get("subgoals")
        if isinstance(raw_subgoals, (str, bytes)) or not isinstance(raw_subgoals, Sequence):
            raise ValueError(f"Episode {episode_idx} subgoals must be a sequence.")
        subgoals = []
        seen_incomplete = False
        for raw_subgoal in raw_subgoals:
            if not isinstance(raw_subgoal, Mapping):
                raise ValueError(f"Episode {episode_idx} contains a non-object subgoal entry.")
            subgoal_id = str(raw_subgoal.get("id", "")).strip()
            label = str(raw_subgoal.get("label", "")).strip()
            skill = str(raw_subgoal.get("skill", "")).strip()
            occurrence = raw_subgoal.get("occurrence")
            achieved = raw_subgoal.get("achieved")
            first_step = raw_subgoal.get("first_achievement_step")
            if not subgoal_id or not label or (not skill):
                raise ValueError(
                    f"Episode {episode_idx} subgoals require non-empty id, label, and skill fields."
                )
            if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
                raise ValueError(
                    f"Episode {episode_idx} subgoal {subgoal_id} has invalid occurrence={occurrence!r}."
                )
            if not isinstance(achieved, bool):
                raise ValueError(
                    f"Episode {episode_idx} subgoal {subgoal_id} achieved must be boolean."
                )
            if achieved:
                if seen_incomplete:
                    raise ValueError(
                        f"Episode {episode_idx} violates ordered subgoal semantics at {subgoal_id}."
                    )
                if (
                    not isinstance(first_step, int)
                    or isinstance(first_step, bool)
                    or first_step < 0
                ):
                    raise ValueError(
                        f"Episode {episode_idx} achieved subgoal {subgoal_id} requires a non-negative first_achievement_step, got {first_step!r}."
                    )
                achieved_steps.setdefault(subgoal_id, []).append(first_step)
            else:
                seen_incomplete = True
                if first_step is not None:
                    raise ValueError(
                        f"Episode {episode_idx} unachieved subgoal {subgoal_id} must have a null first_achievement_step."
                    )
            expected_metadata.setdefault(subgoal_id, (label, skill, occurrence))
            if expected_metadata[subgoal_id] != (label, skill, occurrence):
                raise ValueError(f"Subgoal metadata changed across episodes for {subgoal_id}.")
            subgoals.append(
                {
                    "id": subgoal_id,
                    "label": label,
                    "skill": skill,
                    "occurrence": occurrence,
                    "achieved": achieved,
                    "firstAchievementStep": first_step,
                }
            )
        subgoal_ids = [item["id"] for item in subgoals]
        if len(set(subgoal_ids)) != len(subgoal_ids):
            raise ValueError(
                f"Episode {episode_idx} contains duplicate subgoal IDs: {subgoal_ids}."
            )
        if expected_ids is None:
            expected_ids = subgoal_ids
        elif subgoal_ids != expected_ids:
            raise ValueError(
                f"Episode {episode_idx} subgoal order differs from previous episodes: {subgoal_ids}."
            )
        completed = sum(item["achieved"] for item in subgoals)
        total = len(subgoals)
        first_incomplete = next((item["id"] for item in subgoals if not item["achieved"]), None)
        if record.get("completed_subgoals") != completed or record.get("total_subgoals") != total:
            raise ValueError(
                f"Episode {episode_idx} subgoal counts do not match its milestone entries."
            )
        if record.get("first_incomplete_subgoal") != first_incomplete:
            raise ValueError(
                f"Episode {episode_idx} first_incomplete_subgoal does not match ordered milestones."
            )
        episodes_by_index[episode_idx] = {
            "completed": completed,
            "total": total,
            "firstIncomplete": first_incomplete,
            "subgoals": subgoals,
        }
    if set(episodes_by_index) != set(episode_indices):
        raise ValueError(
            f"Subgoal diagnostic episode indices do not match evaluation episode indices: diagnostics={sorted(episodes_by_index)} evaluation={sorted(episode_indices)}."
        )
    raw_aggregate = diagnostics.get("aggregate")
    if not isinstance(raw_aggregate, Mapping):
        raise ValueError("RoboCasa subgoal_diagnostics.aggregate must be an object.")
    if set(raw_aggregate) != set(expected_ids or []):
        raise ValueError(
            "RoboCasa subgoal diagnostic aggregate IDs do not match episode milestone IDs."
        )
    aggregate = []
    for subgoal_id in expected_ids or []:
        raw_metric = raw_aggregate[subgoal_id]
        if not isinstance(raw_metric, Mapping):
            raise ValueError(f"Aggregate subgoal metric {subgoal_id} must be an object.")
        (label, skill, occurrence) = expected_metadata[subgoal_id]
        steps = achieved_steps.get(subgoal_id, [])
        achieved_count = len(steps)
        expected_average = sum(steps) / len(steps) if steps else None
        if (
            raw_metric.get("label") != label
            or raw_metric.get("skill") != skill
            or raw_metric.get("occurrence") != occurrence
            or (raw_metric.get("achieved_count") != achieved_count)
        ):
            raise ValueError(f"Aggregate subgoal metric {subgoal_id} does not match episode data.")
        rate = raw_metric.get("achievement_rate")
        expected_rate = achieved_count / max(len(episode_indices), 1)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or abs(rate - expected_rate) > 1e-09
        ):
            raise ValueError(
                f"Aggregate subgoal metric {subgoal_id} has an invalid achievement_rate."
            )
        average = raw_metric.get("avg_first_achievement_step")
        if expected_average is None:
            if average is not None:
                raise ValueError(
                    f"Aggregate subgoal metric {subgoal_id} must have a null average step."
                )
        elif (
            not isinstance(average, (int, float))
            or isinstance(average, bool)
            or abs(average - expected_average) > 1e-09
        ):
            raise ValueError(f"Aggregate subgoal metric {subgoal_id} has an invalid average step.")
        aggregate.append(
            {
                "id": subgoal_id,
                "label": label,
                "achievedCount": achieved_count,
                "achievementRate": float(rate),
                "avgFirstAchievementStep": average,
            }
        )
    return {"mode": "ordered_ever", "episodesByIndex": episodes_by_index, "aggregate": aggregate}


def _build_page_data(output_dir: Path, results: Mapping[str, Any]) -> dict[str, Any]:
    task_name = str(results.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("RoboCasa results must contain a non-empty task_name.")
    episode_indices = [int(value) for value in _required_sequence(results, "episode_indices")]
    episode_lengths = [int(value) for value in _required_sequence(results, "episode_lengths")]
    task_descriptions = [str(value) for value in _required_sequence(results, "task_descriptions")]
    reset_attempt_counts_raw = results.get("reset_attempt_counts")
    if reset_attempt_counts_raw is None:
        reset_attempt_counts = [1] * len(episode_indices)
    else:
        reset_attempt_counts = [
            int(value) for value in _required_sequence(results, "reset_attempt_counts")
        ]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"RoboCasa results contain duplicate episode indices: {episode_indices}")
    if (
        len(episode_lengths) != len(episode_indices)
        or len(task_descriptions) != len(episode_indices)
        or len(reset_attempt_counts) != len(episode_indices)
    ):
        raise ValueError(
            f"RoboCasa result episode arrays must have equal lengths: indices={len(episode_indices)} lengths={len(episode_lengths)} descriptions={len(task_descriptions)} reset_attempts={len(reset_attempt_counts)}"
        )
    if any(value < 1 for value in reset_attempt_counts):
        raise ValueError(
            f"RoboCasa reset_attempt_counts must all be at least 1: {reset_attempt_counts}"
        )
    if int(results.get("num_trials", -1)) != len(episode_indices):
        raise ValueError(
            f"RoboCasa results num_trials={results.get('num_trials')} does not match episode count={len(episode_indices)}."
        )
    success_episodes = {int(value) for value in _required_sequence(results, "success_episodes")}
    failure_episodes = {int(value) for value in _required_sequence(results, "failure_episodes")}
    expected_episodes = set(episode_indices)
    if success_episodes & failure_episodes:
        raise ValueError("RoboCasa success_episodes and failure_episodes overlap.")
    if success_episodes | failure_episodes != expected_episodes:
        raise ValueError(
            f"RoboCasa success/failure episode partition does not match episode_indices: indices={sorted(expected_episodes)} successes={sorted(success_episodes)} failures={sorted(failure_episodes)}"
        )
    if int(results.get("successes", -1)) != len(success_episodes):
        raise ValueError(
            f"RoboCasa results successes={results.get('successes')} does not match success_episodes={len(success_episodes)}."
        )
    env_kwargs = results.get("env_kwargs")
    robot = str(env_kwargs.get("robots", "")) if isinstance(env_kwargs, Mapping) else ""
    subgoal_page_data = _build_subgoal_page_data(results, episode_indices)
    episodes = []
    missing_videos = []
    for sequence_index, (episode_index, episode_length, description, reset_attempts) in enumerate(
        zip(episode_indices, episode_lengths, task_descriptions, reset_attempt_counts, strict=True),
        start=1,
    ):
        success = episode_index in success_episodes
        status = "success" if success else "failure"
        video_relative_path = Path("videos") / f"trial{episode_index:03d}_{status}.mp4"
        if not (output_dir / video_relative_path).is_file():
            missing_videos.append(str(output_dir / video_relative_path))
        episode_subgoals = (
            None
            if subgoal_page_data is None
            else subgoal_page_data["episodesByIndex"][episode_index]
        )
        episodes.append(
            {
                "sequence": sequence_index,
                "episodeIndex": episode_index,
                "steps": episode_length,
                "seconds": episode_length / VIDEO_FPS,
                "success": success,
                "description": description,
                "resetAttempts": reset_attempts,
                "video": video_relative_path.as_posix(),
                "subgoalDiagnostics": episode_subgoals,
            }
        )
    if missing_videos:
        raise FileNotFoundError(
            f"RoboCasa result gallery is missing expected video files: {missing_videos}"
        )
    return {
        "taskName": task_name,
        "numTrials": len(episode_indices),
        "successes": len(success_episodes),
        "successRate": float(results.get("success_rate", 0.0)),
        "retriedEpisodes": sum(value > 1 for value in reset_attempt_counts),
        "checkpoint": str(results.get("checkpoint_path", results.get("ckpt", ""))),
        "checkpointType": str(results.get("checkpoint_type", "")),
        "robot": robot,
        "actionHorizon": results.get("action_horizon"),
        "replanSteps": results.get("replan_steps"),
        "numInferenceSteps": results.get("num_inference_steps"),
        "subgoalDiagnostics": None
        if subgoal_page_data is None
        else {"mode": subgoal_page_data["mode"], "aggregate": subgoal_page_data["aggregate"]},
        "episodes": episodes,
    }


def _safe_json_for_script(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


_HTML_TEMPLATE = '<!doctype html>\n<html lang="en">\n<head>\n  <meta charset="utf-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1">\n  <title>RoboCasa Eval Results</title>\n  <style>\n    :root { color-scheme: light; --bg: #f5f6f7; --panel: #fff; --ink: #202427; --muted: #65717a; --line: #d7dde2; --soft: #edf1f4; --accent: #146c8e; --accent2: #b85c24; --ok: #1d7a52; --bad: #a13a32; }\n    * { box-sizing: border-box; }\n    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.42; }\n    header { position: sticky; top: 0; z-index: 20; border-bottom: 1px solid var(--line); background: rgba(245, 246, 247, .96); backdrop-filter: blur(10px); }\n    .topbar { max-width: 1480px; margin: 0 auto; padding: 14px 18px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; align-items: center; }\n    h1 { margin: 0; font-size: 20px; line-height: 1.2; }\n    .subhead { margin: 4px 0 0; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }\n    .subhead strong { color: var(--ink); }\n    .toolbar { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }\n    button { min-height: 36px; border: 1px solid var(--line); border-radius: 6px; padding: 0 12px; background: var(--panel); color: var(--ink); font: inherit; font-size: 13px; cursor: pointer; }\n    button:hover { border-color: var(--accent); }\n    button.primary { border-color: var(--accent); background: var(--accent); color: #fff; }\n    main { max-width: 1480px; margin: 0 auto; padding: 18px; }\n    .diagnostics { margin: 0 0 18px; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }\n    .diagnostics h2 { margin: 0 0 10px; font-size: 17px; }\n    .diagnostics table { width: 100%; border-collapse: collapse; font-size: 13px; }\n    .diagnostics th, .diagnostics td { padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: left; }\n    .diagnostics th { color: var(--muted); font-weight: 600; }\n    .episode-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }\n    .episode { overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }\n    .episode-header { padding: 14px; border-bottom: 1px solid var(--line); background: #fbfcfd; }\n    .episode-title { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin: 0 0 8px; font-size: 17px; }\n    .episode-title span { color: var(--muted); font-size: 13px; }\n    .details { margin: 0; padding: 0; display: grid; gap: 5px; list-style: none; color: #384249; font-size: 13px; }\n    .details li { min-width: 0; overflow-wrap: anywhere; }\n    .details strong { color: var(--muted); }\n    .meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }\n    .pill { display: inline-flex; align-items: center; min-height: 26px; border: 1px solid var(--line); border-radius: 999px; padding: 0 9px; background: var(--soft); color: var(--muted); font-size: 12px; }\n    .pill.ok { border-color: rgba(29, 122, 82, .28); background: rgba(29, 122, 82, .1); color: var(--ok); }\n    .pill.fail { border-color: rgba(161, 58, 50, .28); background: rgba(161, 58, 50, .1); color: var(--bad); }\n    .subgoals { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }\n    .video-wrap { background: #070a0c; }\n    video { display: block; width: 100%; aspect-ratio: 1 / 1; background: #070a0c; object-fit: contain; cursor: pointer; }\n    .seek-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 10px; align-items: center; min-height: 42px; padding: 0 10px; background: #10171b; color: #d9e2e7; font-size: 12px; }\n    .seek-row button { min-height: 28px; border-color: #344550; background: #1b2730; color: #e8eef2; font-size: 12px; }\n    .seek { width: 100%; accent-color: var(--accent2); cursor: pointer; }\n    .time { min-width: 88px; text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #c6d0d7; }\n    @media (max-width: 760px) { .episode-grid, .topbar { grid-template-columns: 1fr; } .toolbar { justify-content: flex-start; } main { padding: 12px; } }\n  </style>\n</head>\n<body>\n  <header><div class="topbar"><div><h1 id="title"></h1><p class="subhead" id="summary"></p><p class="subhead"><strong>Stage 2 checkpoint</strong> <span id="checkpoint"></span></p></div><div class="toolbar"><button class="primary" id="playVisible" type="button">Play visible</button><button id="pauseAll" type="button">Pause all</button><button id="resetAll" type="button">Reset</button></div></div></header>\n  <main><section id="subgoalSummary" class="diagnostics" aria-label="Subgoal achievement summary" hidden><h2>Ordered subgoal achievements</h2><table><thead><tr><th>Subgoal</th><th>Achieved</th><th>Rate</th><th>Average first step</th></tr></thead><tbody id="subgoalSummaryBody"></tbody></table></section><section id="episodes" class="episode-grid" aria-label="RoboCasa evaluation videos"></section></main>\n  <script>\n    const DATA = __PAGE_DATA__;\n    const escapeHtml = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll(\'"\', "&quot;");\n    const secondsLabel = (value) => `${Number(value).toFixed(2)}s`;\n    const task = DATA.taskName;\n    document.title = `RoboCasa ${task} Eval Results`;\n    document.getElementById("title").textContent = `RoboCasa ${task} DreamFree Stage 2 Eval Results`;\n    const retrySummary = DATA.retriedEpisodes > 0 ? ` ${DATA.retriedEpisodes} episode${DATA.retriedEpisodes === 1 ? "" : "s"} required a validated reset retry.` : "";\n    const diagnosticSummary = DATA.subgoalDiagnostics ? " Ordered subgoal achievements are recorded passively." : "";\n    document.getElementById("summary").textContent = `${DATA.numTrials} trials, ${DATA.successes} successful (${(DATA.successRate * 100).toFixed(1)}%), shown as one evaluation video per trial.${retrySummary}${diagnosticSummary}`;\n    document.getElementById("checkpoint").textContent = DATA.checkpoint;\n    if (DATA.subgoalDiagnostics) {\n      const summarySection = document.getElementById("subgoalSummary");\n      const summaryBody = document.getElementById("subgoalSummaryBody");\n      for (const metric of DATA.subgoalDiagnostics.aggregate) {\n        const row = document.createElement("tr");\n        const average = metric.avgFirstAchievementStep === null ? "—" : Number(metric.avgFirstAchievementStep).toFixed(1);\n        row.innerHTML = `<td>${escapeHtml(metric.label)}</td><td>${metric.achievedCount}/${DATA.numTrials}</td><td>${(metric.achievementRate * 100).toFixed(1)}%</td><td>${average}</td>`;\n        summaryBody.append(row);\n      }\n      summarySection.hidden = false;\n    }\n    const root = document.getElementById("episodes");\n    for (const item of DATA.episodes) {\n      const trial = `trial${String(item.episodeIndex).padStart(3, "0")}`;\n      const article = document.createElement("article");\n      article.className = "episode";\n      const subgoalMarkup = item.subgoalDiagnostics ? `<div class="subgoals" aria-label="Episode subgoal achievements">${item.subgoalDiagnostics.subgoals.map((subgoal) => `<span class="pill ${subgoal.achieved ? "ok" : "fail"}">${subgoal.achieved ? "✓" : "✗"} ${escapeHtml(subgoal.label)}${subgoal.achieved ? ` @ step ${subgoal.firstAchievementStep}` : ""}</span>`).join("")}</div>` : "";\n      article.innerHTML = `<div class="episode-header"><h2 class="episode-title">#${String(item.sequence).padStart(2, "0")} ${escapeHtml(task)} <span>${trial}</span></h2><ul class="details"><li><strong>task</strong> ${escapeHtml(item.description)}</li><li><strong>video</strong> ${escapeHtml(item.video)}</li></ul><div class="meta"><span class="pill">${item.steps} steps</span><span class="pill">${secondsLabel(item.seconds)}</span><span class="pill ${item.success ? "ok" : "fail"}">${item.success ? "success" : "failure"}</span>${item.resetAttempts > 1 ? `<span class="pill">reset attempts ${item.resetAttempts}</span>` : ""}${DATA.robot ? `<span class="pill">${escapeHtml(DATA.robot)}</span>` : ""}</div>${subgoalMarkup}</div><div class="video-wrap"><video muted playsinline preload="metadata" src="${escapeHtml(item.video)}" aria-label="${trial} evaluation video"></video><div class="seek-row"><button type="button" class="clip-play">Play</button><input class="seek" type="range" min="0" max="${item.seconds}" value="0" step="0.01" aria-label="Seek ${trial}"><span class="time">0.00 / ${secondsLabel(item.seconds)}</span></div></div>`;\n      root.append(article);\n      const video = article.querySelector("video");\n      const play = article.querySelector(".clip-play");\n      const seek = article.querySelector(".seek");\n      const time = article.querySelector(".time");\n      let duration = item.seconds;\n      const update = () => { seek.value = String(video.currentTime || 0); time.textContent = `${(video.currentTime || 0).toFixed(2)} / ${secondsLabel(duration)}`; };\n      const toggle = () => video.paused ? video.play().catch(() => {}) : video.pause();\n      play.addEventListener("click", toggle); video.addEventListener("click", toggle);\n      seek.addEventListener("input", () => { video.currentTime = Number(seek.value); update(); });\n      video.addEventListener("timeupdate", update);\n      video.addEventListener("loadedmetadata", () => { if (Number.isFinite(video.duration) && video.duration > 0) { duration = video.duration; seek.max = String(duration); } update(); });\n      video.addEventListener("play", () => { play.textContent = "Pause"; });\n      video.addEventListener("pause", () => { play.textContent = "Play"; });\n      video.addEventListener("ended", () => { play.textContent = "Play"; });\n    }\n    const videos = () => Array.from(document.querySelectorAll("video"));\n    document.getElementById("playVisible").addEventListener("click", () => { const height = window.innerHeight; for (const video of videos()) { const rect = video.closest(".episode").getBoundingClientRect(); if (rect.bottom > 0 && rect.top < height) video.play().catch(() => {}); } });\n    document.getElementById("pauseAll").addEventListener("click", () => videos().forEach((video) => video.pause()));\n    document.getElementById("resetAll").addEventListener("click", () => videos().forEach((video) => { video.pause(); video.currentTime = 0; }));\n  </script>\n</body>\n</html>\n'


def write_robocasa_results_html(output_dir: str | Path, results: Mapping[str, Any]) -> tuple[Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"RoboCasa result output directory does not exist: {output_dir}")
    page_data = _build_page_data(output_dir, results)
    html_text = _HTML_TEMPLATE.replace("__PAGE_DATA__", _safe_json_for_script(page_data))
    output_path = output_dir / OUTPUT_NAME
    temporary_path = output_dir / f".{output_path.name}.tmp.{os.getpid()}"
    temporary_path.write_text(html_text, encoding="utf-8")
    os.replace(temporary_path, output_path)
    return (output_path,)
