"""Reproducible ABC episode selection with disjoint operator/day recording groups."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from dreamfree.data.yam import fingerprint

FPS = 30
REPO_ID = "XDOF/ABC-130k"
POLICY = "abc130k_core_breadth_operator_day_v2"
CATEGORIES = (
    "pick_place",
    "fine_pick_place",
    "folding",
    "insertion_ejection",
    "tool_use",
    "sorting",
    "tying_untying",
)


def task_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def rank(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def recording_group(operator: str, timestamp_ns: int) -> str:
    if not operator.strip() or timestamp_ns <= 0:
        raise ValueError(
            "Operator/day splitting requires operator_id and a positive absolute timestamp."
        )
    day = datetime.fromtimestamp(timestamp_ns / 1e9, timezone.utc).date().isoformat()
    return f"{operator}:{day}"


def held_out(group: str, seed: int, fraction: float) -> bool:
    return int(rank(seed, group), 16) / 2**256 < fraction


def reserve_validation_groups(rows: list[dict], *, seed: int, fraction: float) -> set[str]:
    by_task, by_group = defaultdict(set), defaultdict(set)
    for row in rows:
        by_task[row["task_id"]].add(row["group"])
        by_group[row["group"]].add(row["task_id"])
    reserved = {group for group in by_group if held_out(group, seed, fraction)}
    tasks = sorted(by_task, key=lambda task: (len(by_task[task]), task))
    # Keep at least one training group for every task, including tasks with only one group.
    for task in tasks:
        if not by_task[task] - reserved:
            reserved.remove(min(by_task[task], key=lambda group: rank(seed, group)))
    # Add validation coverage when it does not consume any task's last training group.
    for task in tasks:
        if by_task[task] & reserved:
            continue
        candidates = [
            group
            for group in by_task[task]
            if all(by_task[other] - reserved - {group} for other in by_group[group])
        ]
        if candidates:
            reserved.add(min(candidates, key=lambda group: rank(seed, group)))
    return reserved


def load_taxonomy(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text())
    if set(value["categories"]) != set(CATEGORIES):
        raise ValueError(f"Taxonomy must define all seven categories: {path}")
    result = {}
    for category, names in value["categories"].items():
        for name in names:
            key = task_key(name)
            if key in result:
                raise ValueError(f"Duplicate canonical task in taxonomy: {name}")
            result[key] = category
    return result


def validate_source_row(row: dict) -> None:
    path = PurePosixPath(row["source_mcap"])
    if (
        len(path.parts) != 5
        or path.parts[:2] != ("data", "train")
        or path.name != "episode.mcap"
        or ".." in path.parts
        or path.parts[3] != f"episode_{row['source_episode_id']}"
    ):
        raise ValueError(f"Invalid ABC source path/episode identity: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
        raise ValueError(f"Missing original MCAP SHA256: {path}")
    if row["source_size"] <= 0 or int(row["length"]) < 33:
        raise ValueError(f"Invalid ABC file size/duration: {path}")
    if row["group"] != recording_group(row["operator_id"], row["start_ns"]):
        raise ValueError(f"Incorrect ABC operator/day group: {path}")
    if row["task_id"] != task_key(row["instruction"]):
        raise ValueError(f"Incorrect canonical task: {path}")


def choose_core_tasks(
    rows: list[dict],
    validation: list[dict],
    taxonomy: dict[str, str],
    *,
    count: int,
    seed: int,
    min_episodes: int,
    min_frames: int,
) -> list[str]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    val_tasks = {row["task_id"] for row in validation}
    candidates = {
        task
        for task, episodes in by_task.items()
        if task in val_tasks
        and len(episodes) >= min_episodes
        and sum(row["length"] for row in episodes) >= min_frames
    }
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} tasks have enough training data and held-out groups for "
            f"{count} core tasks; expand the index or explicitly reduce --core-tasks."
        )
    missing = set(CATEGORIES) - {taxonomy[task] for task in candidates}
    if missing:
        raise ValueError(
            f"Core candidates lack categories {sorted(missing)}; inspect the taxonomy/source pool."
        )
    words = {
        task: set(task.split())
        - {"the", "a", "and", "of", "into", "in", "on", "to", "from", "with"}
        for task in candidates
    }
    diversity = {
        task: (
            len({row["operator_id"] for row in by_task[task]}),
            len({row["group"] for row in by_task[task]}),
            rank(seed, task),
        )
        for task in candidates
    }
    selected, category_counts = [], Counter()
    while len(selected) < count:
        category = min(
            {taxonomy[task] for task in candidates}, key=lambda key: (category_counts[key], key)
        )
        previous = [task for task in selected if taxonomy[task] == category]

        def priority(task):
            # Lexical distance avoids selecting only closely named variants; it is not a visual embedding.
            distance = min(
                (
                    1 - len(words[task] & words[other]) / len(words[task] | words[other])
                    for other in previous
                ),
                default=1,
            )
            operators, groups, tie = diversity[task]
            return (-distance, -operators, -groups, tie)

        task = min((task for task in candidates if taxonomy[task] == category), key=priority)
        selected.append(task)
        candidates.remove(task)
        category_counts[category] += 1
    return sorted(selected)


def balanced_sample(
    rows: list[dict],
    budget: int,
    *,
    seed: int,
    initial: list[dict] = (),
    minimums: dict[str, int] | None = None,
) -> list[dict]:
    """Balance task duration, then operator duration and recording-group duration within each task."""
    minimums = minimums or {}
    spent, counts, operator_spent, group_spent = Counter(), Counter(), Counter(), Counter()
    selected_ids = {row["source_episode_id"] for row in initial}
    for row in initial:
        task, group, length = row["task_id"], row["group"], row["length"]
        spent[task] += length
        counts[task] += 1
        operator_spent[task, row["operator_id"]] += length
        group_spent[task, group] += length
    queues = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda item: rank(seed, item["source_episode_id"]), reverse=True):
        if row["source_episode_id"] not in selected_ids:
            queues[row["task_id"]][row["group"]].append(row)
    task_ranks = {task: rank(seed, task) for task in queues}
    group_ranks = {group: rank(seed, group) for groups in queues.values() for group in groups}

    def minimum_cost(task, needed):
        if needed <= 0:
            return 0
        lengths = sorted(row["length"] for group in queues.get(task, {}).values() for row in group)
        if len(lengths) < needed:
            raise ValueError(
                f"Episode budget cannot meet required task coverage/counts: {task} needs "
                f"{needed} more episodes, but only {len(lengths)} remain."
            )
        return sum(lengths[:needed])

    minimum_costs = {
        task: minimum_cost(task, minimum - counts[task]) for task, minimum in minimums.items()
    }
    reserved_frames = sum(minimum_costs.values())
    if reserved_frames > budget:
        raise ValueError(
            f"Episode budget cannot meet required task coverage/counts: minimum complete episodes "
            f"need {reserved_frames / FPS / 3600:.4f} hours, budget is {budget / FPS / 3600:.4f}."
        )
    selected = []
    while queues and budget >= 33:
        task = min(
            queues,
            key=lambda key: (
                counts[key] >= minimums.get(key, 0),
                counts[key] / max(minimums.get(key, 0), 1)
                if counts[key] < minimums.get(key, 0)
                else spent[key],
                task_ranks[key],
            ),
        )
        groups = queues[task]
        group = min(
            groups,
            key=lambda key: (
                operator_spent[task, groups[key][-1]["operator_id"]],
                group_spent[task, key],
                group_ranks[key],
            ),
        )
        row = groups[group].pop()
        if not groups[group]:
            del groups[group]
        if not groups:
            del queues[task]
        length = row["length"]
        next_cost = minimum_cost(task, minimums.get(task, 0) - counts[task] - 1)
        remaining_minimum = reserved_frames - minimum_costs.get(task, 0) + next_cost
        # A random long episode must not consume the time needed for other required demonstrations.
        if length + remaining_minimum > budget:
            continue
        selected.append(row)
        budget -= length
        minimum_costs[task] = next_cost
        reserved_frames = remaining_minimum
        spent[task] += length
        counts[task] += 1
        operator_spent[task, row["operator_id"]] += length
        group_spent[task, group] += length
    short = {
        task: (counts[task], minimum)
        for task, minimum in minimums.items()
        if counts[task] < minimum
    }
    if short:
        raise ValueError(f"Episode budget cannot meet required task coverage/counts: {short}")
    return selected


def build_plan(
    index: dict,
    taxonomy: dict[str, str],
    *,
    train_hours: float = 100,
    val_hours: float = 5,
    breadth_hours: float = 75,
    core_tasks: int = 70,
    min_core_episodes: int = 20,
    holdout_fraction: float = 0.1,
    seed: int = 42,
    excluded_ids: set[str] = frozenset(),
) -> dict:
    for name, value in (
        ("train_hours", train_hours),
        ("val_hours", val_hours),
        ("breadth_hours", breadth_hours),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if train_hours + val_hours > 300 or breadth_hours >= train_hours:
        raise ValueError("Require breadth_hours < train_hours and train_hours + val_hours <= 300.")
    if core_tasks < 7 or min_core_episodes < 1 or not 0 < holdout_fraction < 1:
        raise ValueError(
            "Require at least seven core tasks, positive min_core_episodes, and 0 < holdout_fraction < 1."
        )
    known_ids = {
        Path(row["source_mcap"]).parent.name.removeprefix("episode_") for row in index["episodes"]
    }
    if excluded_ids - known_ids:
        raise ValueError(
            f"Excluded episode UUIDs are absent from the source index: {sorted(excluded_ids - known_ids)}"
        )
    rows = [
        row
        for row in index["episodes"]
        if row.get("eligible") and row["source_episode_id"] not in excluded_ids
    ]
    ids = [row["source_episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Source index contains duplicate episode IDs.")
    for row in rows:
        validate_source_row(row)
    unknown = {row["task_id"] for row in rows} - taxonomy.keys()
    if unknown:
        raise ValueError(
            f"Tasks missing explicit category assignments; update --taxonomy: {sorted(unknown)}"
        )
    reserved = reserve_validation_groups(rows, seed=seed, fraction=holdout_fraction)
    train_pool = [row for row in rows if row["group"] not in reserved]
    val_pool = [row for row in rows if row["group"] in reserved]
    frames = lambda hours: int(hours * 3600 * FPS)
    core = choose_core_tasks(
        train_pool,
        val_pool,
        taxonomy,
        count=core_tasks,
        seed=seed,
        min_episodes=min_core_episodes,
        min_frames=frames((train_hours - breadth_hours) / core_tasks),
    )
    train_tasks = {row["task_id"] for row in train_pool}
    val_tasks = train_tasks & {row["task_id"] for row in val_pool}
    # Reserve held-out groups before selecting any training episode. Unselected held-out episodes stay excluded.
    validation = balanced_sample(
        [row for row in val_pool if row["task_id"] in val_tasks],
        frames(val_hours),
        seed=seed,
        minimums={task: 1 for task in val_tasks},
    )
    broad = balanced_sample(
        train_pool, frames(breadth_hours), seed=seed, minimums={task: 1 for task in train_tasks}
    )
    depth = balanced_sample(
        [row for row in train_pool if row["task_id"] in core],
        frames(train_hours) - sum(row["length"] for row in broad),
        seed=seed,
        initial=broad,
        minimums={task: min_core_episodes for task in core},
    )
    selected = [
        {**row, "split": split, "selection_phase": phase, "category": taxonomy[row["task_id"]]}
        for split, phase, group in (
            ("train", "breadth", broad),
            ("train", "core", depth),
            ("val", "validation", validation),
        )
        for row in group
    ]
    for split, hours in (("train", train_hours), ("val", val_hours)):
        actual = sum(row["length"] for row in selected if row["split"] == split) / FPS / 3600
        if actual < hours * 0.995:
            raise ValueError(
                f"Insufficient {split} selection: {actual:.3f}/{hours} hours (must fill at least 99.5%)."
            )
    report = {}
    for split in ("train", "val"):
        group = [row for row in selected if row["split"] == split]
        tasks = defaultdict(lambda: {"episodes": 0, "hours": 0.0})
        for row in group:
            tasks[row["task_id"]]["episodes"] += 1
            tasks[row["task_id"]]["hours"] += row["length"] / FPS / 3600
        report[split] = {
            "episodes": len(group),
            "hours": sum(row["length"] for row in group) / FPS / 3600,
            "tasks": dict(sorted(tasks.items())),
            "groups": len({row["group"] for row in group}),
            "operators": len({row["operator_id"] for row in group}),
            "download_bytes": sum(row["source_size"] for row in group),
        }
        for label, subset in (
            ("core", [row for row in group if row["task_id"] in core]),
            ("non_core", [row for row in group if row["task_id"] not in core]),
        ):
            report[split][label] = {
                "episodes": len(subset),
                "tasks": len({row["task_id"] for row in subset}),
                "hours": sum(row["length"] for row in subset) / FPS / 3600,
            }
    plan = {
        "schema_version": 1,
        "policy": POLICY,
        "repo_id": REPO_ID,
        "revision": index["revision"],
        "index_sha256": fingerprint(index),
        "taxonomy_sha256": fingerprint(taxonomy),
        "seed": seed,
        "train_hours": train_hours,
        "val_hours": val_hours,
        "breadth_hours": breadth_hours,
        "holdout_fraction": holdout_fraction,
        "min_core_episodes": min_core_episodes,
        "held_out_groups": sorted(reserved),
        "core_tasks": core,
        "excluded_ids": sorted(excluded_ids),
        "report": report,
        "tasks_without_training_groups": sorted({row["task_id"] for row in rows} - train_tasks),
        "tasks_without_validation_groups": sorted(train_tasks - val_tasks),
        "episodes": sorted(selected, key=lambda row: row["source_mcap"]),
    }
    plan["selection_sha256"] = fingerprint(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "selection_sha256"}
    if (
        plan.get("schema_version") != 1
        or plan.get("policy") != POLICY
        or plan.get("repo_id") != REPO_ID
        or not re.fullmatch(r"[0-9a-f]{40}", plan.get("revision", ""))
        or plan.get("selection_sha256") != fingerprint(unsigned)
    ):
        raise ValueError("Invalid ABC download selection identity/schema/revision.")
    seen, groups, frames = set(), defaultdict(set), Counter()
    for row in plan["episodes"]:
        validate_source_row(row)
        if row["source_episode_id"] in seen or row["split"] not in {"train", "val"}:
            raise ValueError("Duplicate episode or invalid split in ABC selection.")
        seen.add(row["source_episode_id"])
        split = row["split"]
        if (row["group"] in plan["held_out_groups"]) != (split == "val"):
            raise ValueError(
                f"ABC selection violates reserved operator/day split: {row['source_episode_id']}"
            )
        groups[split].add(row["group"])
        frames[split] += row["length"]
    if not groups["train"] or not groups["val"] or groups["train"] & groups["val"]:
        raise ValueError("ABC train/validation groups are empty or overlap.")
    for split in ("train", "val"):
        cap = int(plan[f"{split}_hours"] * 3600 * FPS)
        if not cap * 0.995 <= frames[split] <= cap:
            raise ValueError(f"ABC {split} selection does not satisfy its hour budget.")


def load_plan(path: Path) -> dict:
    value = json.loads(path.read_text())
    validate_plan(value)
    return value
