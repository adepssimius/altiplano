"""Scheduling: re-dating a project's open tasks from their blocking relations.

`reschedule_project` is one tool built from two halves. `plan` is pure: it takes the
tasks as Vikunja returns them and computes dates, the critical path, and label
changes, with no network access. The tool reads every task in one paginated listing,
hands them to `plan`, and writes only the differences. The tests drive `plan`
directly, with no fake server in the way.

The model, in the order `plan` applies it:

- A task's duration comes from `Estimate: N days` in its description. Anything else
  is one day. `Estimate: part of ...` marks a task that shares its parent's days.
- Blocking relations are finish-to-start. A task starts the day after its latest
  unfinished blocker ends, and never before the start date. `Not before:
  YYYY-MM-DD` in a description is a later floor for that task and its subtasks. A done task keeps its
  dates and constrains nothing.
- A parent's blockers also hold back every subtask beneath it. A parent with open
  subtasks then spans them, from the earliest start to the latest end.
- A backward pass from the project finish gives every task its slack. Zero slack is
  critical.

Days are calendar days. A task on days D1 to Dn is written as D1 09:00 to Dn 17:00
local time. Vikunja's Gantt chart draws the end date's day inclusively, and a task
ending at midnight would draw one day short.
"""

import html
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from altiplano.api import _NO_DATE, _items, _request, _verb
from altiplano.app import mcp
from altiplano.tools.tasks import _write_task

_TAGS = re.compile(r"<[^>]+>")
_ESTIMATE_DAYS = re.compile(r"Estimate:\s*(\d+)\s*days?", re.IGNORECASE)
_ESTIMATE_PART_OF = re.compile(r"Estimate:\s*part of\b", re.IGNORECASE)
_NOT_BEFORE = re.compile(r"Not before:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

_DAY_START = time(9, 0)
_DAY_END = time(17, 0)

# Vikunja honours a per_page this large on v2, and the listing then needs one
# request for any realistic project. The loop below still pages, for an instance
# configured with a lower cap.
_PAGE_SIZE = 500


class ScheduleError(ValueError):
    """The project cannot be scheduled as it stands. Nothing is written."""


@dataclass
class Change:
    """One task whose dates move."""

    task_id: int
    old: tuple[date | None, date | None]
    new: tuple[date, date]
    start: str
    end: str


@dataclass
class Plan:
    """Everything `reschedule_project` would write, and what it reports."""

    finish: date | None
    critical_path: list[int]
    changes: list[Change]
    labels_added: dict[int, list[int]]
    labels_removed: dict[int, list[int]]
    unchanged: int
    done: int
    floors: dict[int, date] = field(default_factory=dict)
    excluded: dict[int, str] = field(default_factory=dict)
    titles: dict[int, str] = field(default_factory=dict)
    identifiers: dict[int, str] = field(default_factory=dict)
    dates: dict[int, tuple[date, date]] = field(default_factory=dict)


def _plain(description: str | None) -> str:
    """The description as text. Vikunja stores it as HTML."""
    return html.unescape(_TAGS.sub(" ", description or ""))


def _parse_duration(description: str | None) -> int | None:
    """Days of work, or None for a task that shares its parent's days."""
    text = _plain(description)
    if _ESTIMATE_PART_OF.search(text):
        return None
    m = _ESTIMATE_DAYS.search(text)
    return max(1, int(m.group(1))) if m else 1


def _parse_floor(description: str | None) -> date | None:
    """The `Not before: YYYY-MM-DD` date in a description, if it has one."""
    m = _NOT_BEFORE.search(_plain(description))
    return date.fromisoformat(m.group(1)) if m else None


def _parse_instant(value: str | None) -> datetime | None:
    if not value or value == _NO_DATE:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _local_day(value: str | None, tz: ZoneInfo) -> date | None:
    instant = _parse_instant(value)
    return instant.astimezone(tz).date() if instant else None


def _stamp(day: date, at: time, tz: ZoneInfo) -> datetime:
    """A wall-clock time on a day, with that day's own UTC offset."""
    return datetime.combine(day, at, tzinfo=tz)


def _related_ids(task: dict, kind: str) -> list[int]:
    return [r["id"] for r in (task.get("related_tasks") or {}).get(kind) or []]


def _related_open(task: dict, kind: str) -> bool:
    return any(not r.get("done") for r in (task.get("related_tasks") or {}).get(kind) or [])


def _find_cycle(nodes: set[int], edges: dict[int, set[int]]) -> list[int]:
    """One cycle among `nodes`, as a closed path.

    `nodes` is what a topological sort could not place: every one is on a cycle or
    downstream of one, and every edge out of them stays among them.
    """
    colour: dict[int, int] = {}
    stack: list[int] = []

    def visit(n: int) -> list[int] | None:
        colour[n] = 1
        stack.append(n)
        for m in sorted(edges.get(n, ())):
            if colour.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in colour:
                found = visit(m)
                if found:
                    return found
        stack.pop()
        colour[n] = 2
        return None

    for n in sorted(nodes):
        if n not in colour:
            found = visit(n)
            if found:
                return found
    return sorted(nodes)  # pragma: no cover - Kahn leaves nodes only when a cycle exists


def plan(
    tasks: list[dict],
    start: date,
    tz: ZoneInfo,
    ready_label_id: int | None = None,
    critical_label_id: int | None = None,
    exclude_label_id: int | None = None,
) -> Plan:
    """Compute the schedule for one project's tasks. Pure: no network access.

    Raises ScheduleError when the relations form a cycle. Its message names the
    tasks on it.
    """
    by_id = {t["id"]: t for t in tasks}
    title = {i: t.get("title") or "" for i, t in by_id.items()}
    ident = {i: t.get("identifier") or f"#{i}" for i, t in by_id.items()}
    is_open = {i for i, t in by_id.items() if not t.get("done")}

    # Relations, restricted to tasks in this project. Both directions of each kind
    # are read, so a relation recorded on one side only still counts.
    blockers: dict[int, set[int]] = {i: set() for i in by_id}
    children: dict[int, set[int]] = {i: set() for i in by_id}
    parent: dict[int, int] = {}
    for i, t in by_id.items():
        for j in _related_ids(t, "blocking"):
            if j in by_id:
                blockers[j].add(i)
        for j in _related_ids(t, "blocked"):
            if j in by_id:
                blockers[i].add(j)
        for j in _related_ids(t, "subtask"):
            if j in by_id:
                children[i].add(j)
        for j in _related_ids(t, "parenttask"):
            if j in by_id:
                children[j].add(i)
    for p, kids in children.items():
        for c in kids:
            if c in parent and parent[c] != p:
                raise ScheduleError(
                    f"{ident[c]} {title[c]} has two parents, {ident[parent[c]]} and {ident[p]}."
                )
            parent[c] = p

    # A subtree that loops back on itself has no span to compute, and walking up
    # from any task in it would never stop.
    _check_subtask_cycles(parent, ident, title)

    # Out of scope: open tasks carrying the exclude label, everything beneath them,
    # and everything waiting on them. None of it can be scheduled without the
    # excluded work. It keeps its dates, and from here on counts as not open.
    excluded = _excluded(by_id, is_open, exclude_label_id, blockers, children, ident)
    is_open = is_open - set(excluded)

    duration = {i: _parse_duration(by_id[i].get("description")) for i in is_open}

    # A per-task floor, `Not before: YYYY-MM-DD`. It holds the task and every
    # subtask beneath it, and successors follow from their blockers' end dates.
    floors: dict[int, date] = {}
    for i in is_open:
        try:
            floor = _parse_floor(by_id[i].get("description"))
        except ValueError as err:
            raise ScheduleError(f"{ident[i]} {title[i]} has a Not before date that is not a date: {err}") from err
        if floor is not None:
            floors[i] = floor

    def open_kids(p: int) -> set[int]:
        return {c for c in children[p] if c in is_open}

    # Three roles for an open task. A part-of task copies its parent's dates, when
    # the parent is open. A spanning parent takes its dates from its open subtasks,
    # part-of ones excepted. Everything else is a leaf and gets dates by duration.
    part_of = {i for i in is_open if duration[i] is None and parent.get(i) in is_open}
    spans: dict[int, set[int]] = {}
    for i in is_open:
        kids = open_kids(i) - part_of
        if kids:
            spans[i] = kids
    leaves = is_open - part_of - set(spans)
    for i in leaves:
        if duration[i] is None:
            # Marked part-of, with no open parent to share days with.
            duration[i] = 1


    def ends(i: int) -> set[int]:
        """The leaves whose end dates decide when task `i` ends."""
        if i in part_of:
            return ends(parent[i])
        if i in spans:
            out: set[int] = set()
            for c in spans[i]:
                out |= ends(c)
            return out
        return {i}

    def ancestors(i: int) -> list[int]:
        out = []
        while i in parent:
            i = parent[i]
            out.append(i)
        return out

    # Predecessors of each leaf: the leaves that must finish before it starts. They
    # come from its own unfinished blockers and those of every ancestor.
    #
    # A blocker inside the holder's own subtree is skipped. A subtask that also
    # blocks its parent is common, and redundant: the parent spans its subtasks
    # already. Pushed down to the subtasks, it would make each wait on itself or on
    # its siblings.
    preds: dict[int, set[int]] = {i: set() for i in leaves}
    for i in leaves:
        for holder in [i, *ancestors(i)]:
            for b in blockers[holder]:
                if b in is_open and holder not in [b, *ancestors(b)] and i not in ends(b):
                    preds[i] |= ends(b)
    succs: dict[int, set[int]] = {i: set() for i in leaves}
    for i, ps in preds.items():
        for p in ps:
            succs[p].add(i)

    order = _toposort(leaves, preds, succs, ident, title)

    es: dict[int, date] = {}
    ef: dict[int, date] = {}
    for i in order:
        held = [floors[h] for h in [i, *ancestors(i)] if h in floors]
        first = max([start, *held, *(ef[p] + timedelta(days=1) for p in preds[i])])
        es[i] = first
        ef[i] = first + timedelta(days=duration[i] - 1)

    def span_of(i: int) -> tuple[date, date]:
        if i in part_of:
            return span_of(parent[i])
        if i in spans:
            parts = [span_of(c) for c in spans[i]]
            return min(p[0] for p in parts), max(p[1] for p in parts)
        return es[i], ef[i]

    new_dates = {i: span_of(i) for i in is_open}
    finish = max((d[1] for d in new_dates.values()), default=None)

    # Backward pass over the leaves, in reverse topological order.
    lf: dict[int, date] = {}
    ls: dict[int, date] = {}
    for i in reversed(order):
        latest = min([finish, *(ls[s] - timedelta(days=1) for s in succs[i])])
        lf[i] = latest
        ls[i] = latest - timedelta(days=duration[i] - 1)
    critical_leaves = {i for i in leaves if ls[i] == es[i]}

    def is_critical(i: int) -> bool:
        if i in part_of:
            return is_critical(parent[i])
        if i in spans:
            # Part-of subtasks are left out: they take their criticality from this
            # parent, and asking them would ask the parent again.
            return all(is_critical(c) for c in spans[i])
        return i in critical_leaves

    critical = {i for i in is_open if is_critical(i)}
    critical_path = sorted(critical_leaves, key=lambda i: (es[i], ef[i], i))

    ready = {
        i
        for i in is_open
        if not _related_open(by_id[i], "blocked") and not _related_open(by_id[i], "subtask")
    }

    changes: list[Change] = []
    for i in sorted(is_open, key=lambda i: (new_dates[i][0], new_dates[i][1], i)):
        d1, dn = new_dates[i]
        start_at, end_at = _stamp(d1, _DAY_START, tz), _stamp(dn, _DAY_END, tz)
        t = by_id[i]
        if _parse_instant(t.get("start_date")) == start_at and _parse_instant(t.get("end_date")) == end_at:
            continue
        changes.append(
            Change(
                task_id=i,
                old=(_local_day(t.get("start_date"), tz), _local_day(t.get("end_date"), tz)),
                new=(d1, dn),
                start=start_at.isoformat(),
                end=end_at.isoformat(),
            )
        )

    added: dict[int, list[int]] = {}
    removed: dict[int, list[int]] = {}
    for label_id, wanted in ((ready_label_id, ready), (critical_label_id, critical)):
        if label_id is None:
            continue
        for i, t in by_id.items():
            has = any(lb.get("id") == label_id for lb in t.get("labels") or [])
            if i in wanted and not has:
                added.setdefault(i, []).append(label_id)
            elif i not in wanted and has:
                removed.setdefault(i, []).append(label_id)

    return Plan(
        finish=finish,
        critical_path=critical_path,
        changes=changes,
        labels_added=added,
        labels_removed=removed,
        unchanged=len(is_open) - len(changes),
        excluded=excluded,
        floors=floors,
        done=sum(1 for t in by_id.values() if t.get("done")),
        titles=title,
        identifiers=ident,
        dates=new_dates,
    )


def _excluded(
    by_id: dict[int, dict],
    is_open: set[int],
    label_id: int | None,
    blockers: dict[int, set[int]],
    children: dict[int, set[int]],
    ident: dict[int, str],
) -> dict[int, str]:
    """Open tasks out of scope, each with the reason it is."""
    if label_id is None:
        return {}
    waiting: dict[int, set[int]] = {i: set() for i in by_id}
    for i, bs in blockers.items():
        for b in bs:
            waiting[b].add(i)
    out: dict[int, str] = {}
    queue = sorted(
        i for i in is_open if any(lb.get("id") == label_id for lb in by_id[i].get("labels") or [])
    )
    for i in queue:
        out[i] = "labelled out of scope"
    while queue:
        i = queue.pop(0)
        for c in sorted(children[i]):
            if c in is_open and c not in out:
                out[c] = f"subtask of {ident[i]}"
                queue.append(c)
        for w in sorted(waiting[i]):
            if w in is_open and w not in out:
                out[w] = f"waits on {ident[i]}"
                queue.append(w)
    return out


def _check_subtask_cycles(parent: dict[int, int], ident: dict, title: dict) -> None:
    """Refuse a parent chain that loops back on itself.

    Each task has one parent at most, checked before this runs. A loop is then
    found by walking up from each task until the chain ends or repeats.
    """
    for first in sorted(parent):
        chain = [first]
        j = first
        while j in parent:
            j = parent[j]
            if j in chain:
                loop = chain[chain.index(j):] + [j]
                raise ScheduleError(_cycle_message("subtask", loop[::-1], ident, title))
            chain.append(j)


def _toposort(
    nodes: set[int], preds: dict[int, set[int]], succs: dict[int, set[int]], ident: dict, title: dict
) -> list[int]:
    """Kahn's algorithm, smallest id first among ties so the result is stable."""
    indegree = {i: len(preds[i]) for i in nodes}
    ready = sorted(i for i, d in indegree.items() if d == 0)
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for s in sorted(succs[i]):
            indegree[s] -= 1
            if indegree[s] == 0:
                ready.append(s)
        ready.sort()
    if len(order) < len(nodes):
        left = set(nodes) - set(order)
        raise ScheduleError(_cycle_message("blocking", _find_cycle(left, succs), ident, title))
    return order


def _cycle_message(kind: str, cycle: list[int], ident: dict, title: dict) -> str:
    path = " -> ".join(f"{ident[i]} {title[i]}" for i in cycle)
    return f"The {kind} relations form a cycle, and nothing was written: {path}"


def _fmt_day(d: date | None) -> str:
    return d.isoformat() if d else "unscheduled"


def _fmt_span(span: tuple[date | None, date | None]) -> str:
    a, b = span
    if a is None and b is None:
        return "unscheduled"
    return _fmt_day(a) if a == b else f"{_fmt_day(a)} to {_fmt_day(b)}"


def summarise(p: Plan, label_names: dict[int, str], dry_run: bool) -> str:
    """The compact report the tool returns. No descriptions, no related tasks."""

    def name(i: int) -> str:
        return f"{p.identifiers[i]} {p.titles[i]}"

    lines = ["Dry run. Nothing was written." if dry_run else "Rescheduled."]
    lines.append(f"Project finish: {_fmt_day(p.finish)}")

    lines.append("")
    lines.append(f"Critical path ({len(p.critical_path)}):")
    lines += [f"- {name(i)} — {_fmt_span(p.dates[i])}" for i in p.critical_path] or ["- none"]

    lines.append("")
    lines.append(f"Date changes ({len(p.changes)}):")
    lines += [f"- {name(c.task_id)}: {_fmt_span(c.old)} → {_fmt_span(c.new)}" for c in p.changes] or [
        "- none"
    ]

    if p.floors:
        lines.append("")
        lines.append(f"Held by Not before ({len(p.floors)}):")
        for i in sorted(p.floors, key=lambda i: (p.floors[i], i)):
            lines.append(f"- {name(i)}: not before {p.floors[i].isoformat()}, starts {_fmt_day(p.dates[i][0])}")

    if p.labels_added or p.labels_removed:
        lines.append("")
        lines.append("Labels:")
        for i in sorted(set(p.labels_added) | set(p.labels_removed)):
            bits = [f"+{label_names.get(lb, lb)}" for lb in p.labels_added.get(i, [])]
            bits += [f"-{label_names.get(lb, lb)}" for lb in p.labels_removed.get(i, [])]
            lines.append(f"- {name(i)}: {', '.join(bits)}")

    if p.excluded:
        lines.append("")
        lines.append(f"Not moved, out of scope ({len(p.excluded)}):")
        lines += [f"- {name(i)}: {why}" for i, why in sorted(p.excluded.items())]

    lines.append("")
    lines.append(f"Unchanged: {p.unchanged} open, {p.done} done left as they were.")
    return "\n".join(lines)


async def _project_tasks(project_id: int) -> list[dict]:
    """Every task in the project, done ones included, across as many pages as it takes."""
    out: list[dict] = []
    page = 1
    while True:
        data = await _request(
            "GET", f"/projects/{project_id}/tasks", params={"page": page, "per_page": _PAGE_SIZE}
        )
        batch = _items(data)
        out += batch
        total_pages = data.get("total_pages") if isinstance(data, dict) else None
        if not batch or (total_pages is not None and page >= total_pages) or (
            total_pages is None and len(batch) < _PAGE_SIZE
        ):
            return out
        page += 1


async def _apply(p: Plan) -> None:
    """Write the plan: dates first, then labels. A failure says what already landed."""
    done: list[str] = []
    try:
        for c in p.changes:
            await _write_task(c.task_id, {"start_date": c.start, "end_date": c.end})
            done.append(f"dates on {p.identifiers[c.task_id]}")
        for i, ids in sorted(p.labels_added.items()):
            for label_id in ids:
                await _request(_verb("create"), f"/tasks/{i}/labels", json={"label_id": label_id})
                done.append(f"label {label_id} onto {p.identifiers[i]}")
        for i, ids in sorted(p.labels_removed.items()):
            for label_id in ids:
                await _request("DELETE", f"/tasks/{i}/labels/{label_id}")
                done.append(f"label {label_id} off {p.identifiers[i]}")
    except Exception as err:
        written = ", ".join(done) if done else "nothing"
        raise RuntimeError(
            f"reschedule_project stopped partway. Already written: {written}. "
            f"Re-run it to finish: it computes from current state. Cause: {err}"
        ) from err


@mcp.tool()
async def reschedule_project(
    project_id: int,
    start_date: str | None = None,
    timezone: str = "America/New_York",
    ready_label_id: int | None = None,
    critical_label_id: int | None = None,
    exclude_label_id: int | None = None,
    dry_run: bool = False,
) -> str:
    """Re-date every open task in a project from its blocking relations, in one call.

    Durations come from `Estimate: N days` in each description, one day when there
    is none. `Estimate: part of ...` puts a subtask on its parent's days. Blocking
    relations are finish-to-start, in calendar days: a task starts the day after its
    latest unfinished blocker ends, and no earlier than `start_date` (an ISO date,
    today in `timezone` by default). Done tasks keep their dates and constrain
    nothing. A parent's blockers hold back its subtasks, and a parent with open
    subtasks spans them. Only relations between tasks in this project count.

    `Not before: YYYY-MM-DD` in a task's description holds that task and its
    subtasks to that date or later, and its successors follow. The summary lists
    every task carrying one.

    Dates are written as 09:00 on the first day and 17:00 on the last, local time,
    which Vikunja's Gantt chart draws across exactly those days. Only tasks whose
    dates change are written.

    With `critical_label_id`, that label goes on every task with zero slack and
    comes off every other task in the project. With `ready_label_id`, that label
    goes on every open task with no unfinished blocker or subtask and comes off the
    rest. Resolve both ids with `list_labels` first.

    With `exclude_label_id`, a task carrying that label is out of scope, along
    with its subtasks and every task waiting on it. None of them move, none count
    toward the finish date or the critical path, and they lose the ready and
    critical labels. The summary lists each one and why.

    A cycle in the relations is an error naming the tasks on it, and nothing is
    written. `dry_run` computes the same plan and writes nothing.

    Returns a text summary: the finish date, the critical path, each date change,
    each label change, and a count of tasks left alone.
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as err:
        raise ValueError(f"unknown timezone {timezone!r}") from err
    start = date.fromisoformat(start_date) if start_date else datetime.now(tz).date()

    tasks = await _project_tasks(project_id)
    p = plan(
        tasks,
        start,
        tz,
        ready_label_id=ready_label_id,
        critical_label_id=critical_label_id,
        exclude_label_id=exclude_label_id,
    )

    # Names for the report, taken from tasks that already carry the label. A label
    # on no task yet is reported by id.
    label_names = {lb["id"]: lb.get("title") or str(lb["id"]) for t in tasks for lb in t.get("labels") or []}

    if not dry_run:
        await _apply(p)
    return summarise(p, label_names, dry_run)

