"""reschedule_project: the planner, the report, and the writes.

`plan` is pure, and most of these drive it directly with task dicts shaped the way
the project task listing returns them. The tests at the end go through the tool,
against the fake server, for the reads, the writes, and dry runs.
"""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from altiplano.tools import schedule
from altiplano.tools.schedule import ScheduleError, plan, summarise

NY = ZoneInfo("America/New_York")
START = date(2026, 10, 5)
NO_DATE = "0001-01-01T00:00:00Z"


def task(id, title=None, days=None, desc=None, done=False, labels=(), start=NO_DATE, end=NO_DATE, **rel):
    """A task as the listing returns it. Relation kwargs take lists of task ids.

    Each related entry carries its own `done`, the way Vikunja embeds the related
    task. `done_ids` overrides it for tests that need a related task marked done.
    """
    done_ids = rel.pop("done_ids", set())
    if desc is None:
        desc = f"<p><strong>Estimate:</strong> {days} days (about 3 h)</p>" if days else ""
    return {
        "id": id,
        "identifier": f"#{id}",
        "title": title or f"Task {id}",
        "description": desc,
        "done": done,
        "start_date": start,
        "end_date": end,
        "labels": [{"id": lb, "title": f"L{lb}"} for lb in labels],
        "related_tasks": {
            kind: [{"id": j, "done": j in done_ids, "description": "secret"} for j in ids]
            for kind, ids in rel.items()
        },
    }


def link(tasks, kind, a, b):
    """Record `a <kind> b` on both tasks, as Vikunja does."""
    inverse = {"blocking": "blocked", "subtask": "parenttask"}[kind]
    by = {t["id"]: t for t in tasks}
    by[a]["related_tasks"].setdefault(kind, []).append({"id": b, "done": by[b]["done"]})
    by[b]["related_tasks"].setdefault(inverse, []).append({"id": a, "done": by[a]["done"]})
    return tasks


def days(p, i):
    return p.dates[i]


def d(s):
    return date.fromisoformat(s)


# --- durations ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("<p><strong>Estimate:</strong> 3 days (about 12 h)</p>", 3),
        ("<p>estimate: 1 day</p>", 1),
        ("<p>Estimate:</p><p>2 days</p>", 2),
        ("Estimate: 0 days", 1),
        ("Estimate: 2 days. Later: Estimate: 5 days", 2),
        ("Estimate: 14 working days", 1),
        ("no estimate at all", 1),
        (None, 1),
        ("<p><strong>Estimate:</strong> part of rental day (about 3 h)</p>", None),
        ("Estimate: 2 &amp; a bit", 1),
    ],
)
def test_duration_is_parsed_from_the_html_description(description, expected):
    assert schedule._parse_duration(description) == expected


# --- forward pass ---------------------------------------------------------------
def test_a_simple_chain_runs_end_to_end():
    tasks = link(link([task(1, days=2), task(2, days=1), task(3, days=3)], "blocking", 1, 2), "blocking", 2, 3)
    p = plan(tasks, START, NY)
    assert days(p, 1) == (d("2026-10-05"), d("2026-10-06"))
    assert days(p, 2) == (d("2026-10-07"), d("2026-10-07"))
    assert days(p, 3) == (d("2026-10-08"), d("2026-10-10"))
    assert p.finish == d("2026-10-10")
    assert p.critical_path == [1, 2, 3]


def test_a_task_with_two_blockers_waits_for_the_later_one():
    tasks = [task(1, days=1), task(2, days=4), task(3, days=1)]
    link(tasks, "blocking", 1, 3)
    link(tasks, "blocking", 2, 3)
    p = plan(tasks, START, NY)
    assert days(p, 3) == (d("2026-10-09"), d("2026-10-09"))
    # The short branch has three days of slack and is off the critical path.
    assert p.critical_path == [2, 3]


def test_a_done_blocker_keeps_its_dates_and_constrains_nothing():
    old = "2026-12-01T14:00:00Z"
    tasks = [task(1, done=True, days=5, start=old, end=old), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    p = plan(tasks, START, NY)
    assert days(p, 2) == (START, START)
    assert 1 not in p.dates
    assert [c.task_id for c in p.changes] == [2]
    assert p.done == 1


def test_no_open_task_starts_before_the_start_date():
    p = plan([task(1, days=1, start="2026-09-01T13:00:00Z", end="2026-09-01T21:00:00Z")], START, NY)
    assert days(p, 1) == (START, START)


def test_blockers_outside_the_project_are_ignored():
    p = plan([task(2, days=1, blocked=[99])], START, NY)
    assert days(p, 2) == (START, START)


def test_relation_kinds_other_than_blocking_and_subtask_are_ignored():
    p = plan([task(1, days=2, precedes=[2], related=[2]), task(2, days=1, follows=[1], related=[1])], START, NY)
    assert days(p, 2) == (START, START)


def test_a_relation_recorded_on_one_side_only_still_counts():
    p = plan([task(1, days=2, blocking=[2]), task(2, days=1)], START, NY)
    assert days(p, 2) == (d("2026-10-07"), d("2026-10-07"))
    p = plan([task(1, days=2), task(2, days=1, blocked=[1])], START, NY)
    assert days(p, 2) == (d("2026-10-07"), d("2026-10-07"))


# --- parents and subtasks -----------------------------------------------------------
def test_a_parent_spans_its_subtasks():
    tasks = [task(10, days=9), task(11, days=2), task(12, days=1)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    link(tasks, "blocking", 11, 12)
    p = plan(tasks, START, NY)
    assert days(p, 11) == (d("2026-10-05"), d("2026-10-06"))
    assert days(p, 12) == (d("2026-10-07"), d("2026-10-07"))
    # The parent's own estimate is ignored once it has open subtasks.
    assert days(p, 10) == (d("2026-10-05"), d("2026-10-07"))


def test_a_parents_blockers_hold_back_its_subtasks():
    tasks = [task(1, days=3), task(10), task(11, days=1), task(12, days=1)]
    link(tasks, "blocking", 1, 10)
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    p = plan(tasks, START, NY)
    assert days(p, 11) == days(p, 12) == (d("2026-10-08"), d("2026-10-08"))
    assert days(p, 10) == (d("2026-10-08"), d("2026-10-08"))


def test_a_successor_of_a_parent_waits_for_its_last_subtask():
    tasks = [task(10), task(11, days=1), task(12, days=3), task(20, days=1)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    link(tasks, "blocking", 10, 20)
    p = plan(tasks, START, NY)
    assert days(p, 20) == (d("2026-10-08"), d("2026-10-08"))
    assert p.critical_path == [12, 20]


def test_a_subtask_that_also_blocks_its_parent_is_not_a_cycle():
    """Seen in real data: two subtasks each also recorded as blocking the parent."""
    tasks = [task(8), task(9, days=1), task(10, days=2), task(20, days=1)]
    link(tasks, "subtask", 8, 9)
    link(tasks, "subtask", 8, 10)
    link(tasks, "blocking", 9, 8)
    link(tasks, "blocking", 10, 8)
    link(tasks, "blocking", 8, 20)
    p = plan(tasks, START, NY)
    assert days(p, 9) == (START, START)
    assert days(p, 10) == (d("2026-10-05"), d("2026-10-06"))
    assert days(p, 8) == (d("2026-10-05"), d("2026-10-06"))
    assert days(p, 20) == (d("2026-10-07"), d("2026-10-07"))


def test_a_parent_with_only_done_subtasks_is_scheduled_like_a_leaf():
    tasks = [task(10, days=2), task(11, done=True, days=5)]
    link(tasks, "subtask", 10, 11)
    p = plan(tasks, START, NY)
    assert days(p, 10) == (d("2026-10-05"), d("2026-10-06"))


def test_a_part_of_subtask_shares_its_parents_days():
    part = "<p><strong>Estimate:</strong> part of rental day (about 3 h)</p>"
    tasks = [task(1, days=2), task(36, days=1), task(37, desc=part), task(38, desc=part), task(40, days=1)]
    link(tasks, "blocking", 1, 36)
    link(tasks, "subtask", 36, 37)
    link(tasks, "subtask", 36, 38)
    link(tasks, "blocking", 37, 40)
    p = plan(tasks, START, NY)
    rental = (d("2026-10-07"), d("2026-10-07"))
    assert days(p, 36) == days(p, 37) == days(p, 38) == rental
    # A successor of a part-of task waits for the day it shares.
    assert days(p, 40) == (d("2026-10-08"), d("2026-10-08"))
    assert {36, 37, 38} <= {i for i in p.dates if p.dates[i] == rental}


def test_a_part_of_task_with_no_open_parent_takes_one_day():
    part = "Estimate: part of rental day"
    tasks = [task(36, done=True), task(37, desc=part, days=None)]
    link(tasks, "subtask", 36, 37)
    p = plan(tasks, START, NY)
    assert days(p, 37) == (START, START)


def test_a_task_with_two_parents_is_refused():
    tasks = [task(1), task(2), task(3, days=1)]
    link(tasks, "subtask", 1, 3)
    link(tasks, "subtask", 2, 3)
    with pytest.raises(ScheduleError, match="two parents"):
        plan(tasks, START, NY)


# --- cycles -----------------------------------------------------------------------
def test_a_blocking_cycle_is_an_error_naming_the_tasks_on_it():
    tasks = [task(1, title="Dig"), task(2, title="Fill"), task(3, title="Tamp"), task(4, title="Unrelated")]
    link(tasks, "blocking", 1, 2)
    link(tasks, "blocking", 2, 3)
    link(tasks, "blocking", 3, 1)
    link(tasks, "blocking", 3, 4)
    with pytest.raises(ScheduleError) as err:
        plan(tasks, START, NY)
    message = str(err.value)
    assert "nothing was written" in message
    assert "#1 Dig -> #2 Fill -> #3 Tamp -> #1 Dig" in message
    assert "Unrelated" not in message


def test_a_subtask_cycle_is_an_error():
    tasks = [task(1), task(2), task(3, days=1)]
    link(tasks, "subtask", 1, 2)
    link(tasks, "subtask", 2, 1)
    link(tasks, "subtask", 2, 3)
    with pytest.raises(ScheduleError, match="(subtask relations form a cycle|two parents)"):
        plan(tasks, START, NY)


def test_a_subtask_loop_with_one_parent_each_is_a_subtask_cycle():
    tasks = [task(1), task(2), task(3, days=1), task(4, days=1)]
    link(tasks, "subtask", 1, 2)
    link(tasks, "subtask", 2, 1)
    link(tasks, "subtask", 1, 3)
    link(tasks, "subtask", 2, 4)
    with pytest.raises(ScheduleError, match="subtask relations form a cycle"):
        plan(tasks, START, NY)


def test_cycle_search_steps_past_dead_ends_downstream_of_the_cycle():
    # 10 <-> 11 is the cycle. 1 and 2 sit downstream of it, and the search meets
    # them first.
    edges = {10: {1, 11}, 11: {10}, 1: {2}, 2: set()}
    assert schedule._find_cycle({1, 2, 10, 11}, edges) == [10, 11, 10]
    # The same shape, numbered so the search reaches the cycle first and backs out
    # of a dead end partway along it.
    edges = {1: {5, 2}, 2: {1}, 5: set()}
    assert schedule._find_cycle({1, 2, 5}, edges) == [1, 2, 1]


def test_nested_parents_span_their_subtrees():
    tasks = [task(1), task(2), task(3, days=2), task(4, days=1), task(5, days=4)]
    link(tasks, "subtask", 1, 2)
    link(tasks, "subtask", 2, 3)
    link(tasks, "subtask", 2, 4)
    link(tasks, "subtask", 1, 5)
    link(tasks, "blocking", 3, 4)
    p = plan(tasks, START, NY)
    assert days(p, 2) == (d("2026-10-05"), d("2026-10-07"))
    assert days(p, 1) == (d("2026-10-05"), d("2026-10-08"))


def test_relations_to_tasks_in_other_projects_are_dropped():
    tasks = [task(1, days=1, blocking=[90], subtask=[91], parenttask=[92]), task(2, days=1, blocked=[93])]
    p = plan(tasks, START, NY)
    assert days(p, 1) == days(p, 2) == (START, START)


# --- dates on the wire --------------------------------------------------------------
def test_a_one_day_task_starts_and_ends_on_the_same_date():
    p = plan([task(1, days=1)], START, NY)
    (change,) = p.changes
    assert change.start == "2026-10-05T09:00:00-04:00"
    assert change.end == "2026-10-05T17:00:00-04:00"


def test_dst_ending_on_november_1_changes_the_offset():
    # 2026-11-01 is the first Sunday in November: New York falls back that morning.
    tasks = [task(1, days=2), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    p = plan(tasks, date(2026, 10, 31), NY)
    by = {c.task_id: c for c in p.changes}
    assert by[1].start == "2026-10-31T09:00:00-04:00"
    assert by[1].end == "2026-11-01T17:00:00-05:00"
    assert by[2].start == "2026-11-02T09:00:00-05:00"


def test_dst_starting_in_march_changes_the_offset_the_other_way():
    p = plan([task(1, days=2)], date(2027, 3, 13), NY)
    (change,) = p.changes
    assert change.start == "2027-03-13T09:00:00-05:00"
    assert change.end == "2027-03-14T17:00:00-04:00"


def test_only_tasks_whose_dates_change_are_written():
    same_start = "2026-10-05T13:00:00Z"  # 09:00 in New York
    same_end = "2026-10-05T21:00:00Z"  # 17:00
    tasks = [task(1, days=1, start=same_start, end=same_end), task(2, days=1, start=same_start, end=NO_DATE)]
    p = plan(tasks, START, NY)
    assert [c.task_id for c in p.changes] == [2]
    assert p.unchanged == 1
    (change,) = p.changes
    assert change.old == (START, None)


# --- critical path and labels ---------------------------------------------------------
def test_critical_label_follows_zero_slack_and_comes_off_everything_else():
    tasks = [task(1, days=3), task(2, days=1, labels=[7]), task(3, days=1), task(4, done=True, labels=[7])]
    link(tasks, "blocking", 1, 3)
    link(tasks, "blocking", 2, 3)
    p = plan(tasks, START, NY, critical_label_id=7)
    assert p.labels_added == {1: [7], 3: [7]}
    assert p.labels_removed == {2: [7], 4: [7]}


def test_a_parent_is_critical_only_when_all_its_open_subtasks_are():
    tasks = [task(10), task(11, days=3), task(12, days=1), task(20), task(21, days=3), task(22, days=3)]
    for p_, c in ((10, 11), (10, 12), (20, 21), (20, 22)):
        link(tasks, "subtask", p_, c)
    p = plan(tasks, START, NY, critical_label_id=7)
    assert set(p.labels_added) == {11, 20, 21, 22}


def test_a_part_of_task_is_critical_with_its_parent():
    part = "Estimate: part of rental day"
    tasks = [task(36, days=1), task(37, desc=part), task(50, days=1)]
    link(tasks, "subtask", 36, 37)
    p = plan(tasks, START, NY, critical_label_id=7)
    assert set(p.labels_added) == {36, 37, 50}


def test_ready_label_marks_unblocked_leaves_and_comes_off_the_rest():
    tasks = [
        task(1, days=1),
        task(2, days=1, labels=[5]),
        task(3, days=1, done=True, labels=[5]),
        task(10, labels=[5]),
        task(11, days=1),
        task(20, days=1, blocked=[99], done_ids={99}),
        task(30, days=1, blocked=[98]),
    ]
    link(tasks, "blocking", 1, 2)
    link(tasks, "blocking", 3, 1)
    link(tasks, "subtask", 10, 11)
    p = plan(tasks, START, NY, ready_label_id=5)
    # 1 has only a done blocker. 11 has no blocker of its own. 20's blocker is in
    # another project and done. 30's is in another project and open.
    assert p.labels_added == {1: [5], 11: [5], 20: [5]}
    assert p.labels_removed == {2: [5], 3: [5], 10: [5]}


def test_labels_are_left_alone_when_no_label_id_is_given():
    p = plan([task(1, days=1, labels=[5, 7])], START, NY)
    assert p.labels_added == p.labels_removed == {}


# --- the report -----------------------------------------------------------------------
def test_summary_is_compact_and_carries_no_descriptions():
    tasks = [task(1, title="Dig", days=2, desc="Estimate: 2 days SECRET"), task(2, title="Fill", days=1, labels=[7])]
    link(tasks, "blocking", 1, 2)
    p = plan(tasks, START, NY, critical_label_id=9)
    text = summarise(p, {9: "Critical"}, dry_run=True)
    assert text.splitlines()[:2] == ["Dry run. Nothing was written.", "Project finish: 2026-10-07"]
    assert "- #1 Dig — 2026-10-05 to 2026-10-06" in text
    assert "- #2 Fill — 2026-10-07" in text
    assert "- #1 Dig: unscheduled → 2026-10-05 to 2026-10-06" in text
    assert "- #1 Dig: +Critical" in text
    assert "Unchanged: 0 open, 0 done left as they were." in text
    assert "SECRET" not in text
    assert "secret" not in text


def test_summary_of_an_empty_project():
    text = summarise(plan([], START, NY), {}, dry_run=False)
    assert text.splitlines()[:2] == ["Rescheduled.", "Project finish: unscheduled"]
    assert "Critical path (0):\n- none" in text
    assert "Date changes (0):\n- none" in text
    assert "Labels:" not in text


# --- through the tool ---------------------------------------------------------------------
def chain_page(**envelope):
    tasks = [task(1, days=1, labels=[7]), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    return {"items": tasks, **envelope} if envelope else tasks


def test_dry_run_reads_once_and_writes_nothing(api, run):
    api.returns(chain_page())
    text = run(schedule.reschedule_project(3, start_date="2026-10-05", critical_label_id=9, dry_run=True))
    assert [r.method for r in api.requests] == ["GET"]
    assert api.last.url.path.endswith("/projects/3/tasks")
    assert text.startswith("Dry run. Nothing was written.")
    assert "- #1 Task 1: +9" in text


@pytest.mark.parametrize("api_version", [2])
def test_writes_dates_then_labels(api, run, api_version):
    api.returns_in_order(
        httpx.Response(200, json={"items": chain_page(), "total_pages": 1}),
        httpx.Response(200, json={"id": 1}),
    )
    text = run(schedule.reschedule_project(3, start_date="2026-10-05", ready_label_id=7, critical_label_id=9))
    calls = [(r.method, r.url.path.removeprefix("/api/v2")) for r in api.requests]
    assert calls == [
        ("GET", "/projects/3/tasks"),
        ("PATCH", "/tasks/1"),
        ("PATCH", "/tasks/2"),
        ("POST", "/tasks/1/labels"),
        ("POST", "/tasks/2/labels"),
    ]
    assert json.loads(api.requests[1].content) == {
        "start_date": "2026-10-05T09:00:00-04:00",
        "end_date": "2026-10-05T17:00:00-04:00",
    }
    # Task 1 already carries the ready label, and task 2 waits on task 1. Both are
    # critical.
    labels_sent = [json.loads(r.content)["label_id"] for r in api.requests[3:]]
    assert labels_sent == [9, 9]
    assert text.startswith("Rescheduled.")


@pytest.mark.parametrize("api_version", [2])
def test_removes_labels_with_a_delete(api, run, api_version):
    tasks = [task(1, days=1, done=True, labels=[7])]
    api.returns({"items": tasks, "total_pages": 1})
    run(schedule.reschedule_project(3, start_date="2026-10-05", ready_label_id=7))
    assert [(r.method, r.url.path.removeprefix("/api/v2")) for r in api.requests[1:]] == [
        ("DELETE", "/tasks/1/labels/7")
    ]


@pytest.mark.parametrize("api_version", [2])
def test_reads_every_page_of_a_v2_listing(api, run, api_version):
    api.returns_in_order(
        httpx.Response(200, json={"items": [task(1, days=1)], "total_pages": 2}),
        httpx.Response(200, json={"items": [task(2, days=1)], "total_pages": 2}),
    )
    run(schedule.reschedule_project(3, start_date="2026-10-05", dry_run=True))
    assert [r.url.params["page"] for r in api.requests] == ["1", "2"]


def test_reads_every_page_of_a_v1_listing(api, run, monkeypatch):
    monkeypatch.setattr(schedule, "_PAGE_SIZE", 1)
    api.returns_in_order(
        httpx.Response(200, json=[task(1, days=1)]),
        httpx.Response(200, json=[task(2, days=1)]),
        httpx.Response(200, json=[]),
    )
    text = run(schedule.reschedule_project(3, start_date="2026-10-05", dry_run=True))
    assert [r.url.params["page"] for r in api.requests] == ["1", "2", "3"]
    assert "Date changes (2)" in text


def test_a_cycle_writes_nothing(api, run):
    tasks = [task(1), task(2)]
    link(tasks, "blocking", 1, 2)
    link(tasks, "blocking", 2, 1)
    api.returns(tasks)
    with pytest.raises(ScheduleError, match="cycle"):
        run(schedule.reschedule_project(3, start_date="2026-10-05"))
    assert [r.method for r in api.requests] == ["GET"]


@pytest.mark.parametrize("api_version", [2])
def test_a_failed_write_reports_what_already_landed(api, run, api_version):
    api.returns_in_order(
        httpx.Response(200, json={"items": chain_page(), "total_pages": 1}),
        httpx.Response(200, json={"id": 1}),
        httpx.Response(500, json={"message": "boom"}),
    )
    with pytest.raises(RuntimeError) as err:
        run(schedule.reschedule_project(3, start_date="2026-10-05"))
    message = str(err.value)
    assert "Already written: dates on #1." in message
    assert "boom" in message


@pytest.mark.parametrize("api_version", [2])
def test_a_failure_before_any_write_says_nothing_landed(api, run, api_version):
    api.returns_in_order(
        httpx.Response(200, json={"items": chain_page(), "total_pages": 1}),
        httpx.Response(500, json={"message": "boom"}),
    )
    with pytest.raises(RuntimeError, match="Already written: nothing"):
        run(schedule.reschedule_project(3, start_date="2026-10-05"))


@pytest.mark.parametrize("api_version", [2])
def test_label_writes_are_reported_too(api, run, api_version):
    tasks = [task(1, days=1, start="2026-10-05T13:00:00Z", end="2026-10-05T21:00:00Z")]
    api.returns_in_order(
        httpx.Response(200, json={"items": tasks, "total_pages": 1}),
        httpx.Response(200, json={"ok": True}),
        httpx.Response(500, json={"message": "boom"}),
    )
    with pytest.raises(RuntimeError, match="label 7 onto #1"):
        run(schedule.reschedule_project(3, start_date="2026-10-05", ready_label_id=7, critical_label_id=8))


def test_an_unknown_timezone_is_refused_before_any_request(api, run):
    with pytest.raises(ValueError, match="unknown timezone"):
        run(schedule.reschedule_project(3, timezone="Mars/Olympus_Mons"))
    assert api.requests == []


def test_start_date_defaults_to_today_in_the_timezone(api, run, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            # 02:30 UTC on the 6th is still the 5th in New York.
            return datetime(2026, 10, 6, 2, 30, tzinfo=ZoneInfo("UTC")).astimezone(tz)

    monkeypatch.setattr(schedule, "datetime", Frozen)
    api.returns([task(1, days=1)])
    text = run(schedule.reschedule_project(3, dry_run=True))
    assert "Project finish: 2026-10-05" in text
