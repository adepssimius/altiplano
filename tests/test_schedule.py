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


# --- Not before ----------------------------------------------------------------------
def test_not_before_holds_a_task_and_its_successors():
    tasks = [task(1, desc="Estimate: 2 days<p>Not before: 2026-10-20</p>"), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    p = plan(tasks, START, NY)
    assert days(p, 1) == (d("2026-10-20"), d("2026-10-21"))
    assert days(p, 2) == (d("2026-10-22"), d("2026-10-22"))
    assert p.floors == {1: d("2026-10-20")}


def test_not_before_on_a_parent_holds_its_subtasks():
    tasks = [task(10, desc="not before: 2026-11-02"), task(11, days=1), task(12, days=2)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    p = plan(tasks, START, NY)
    assert days(p, 11) == (d("2026-11-02"), d("2026-11-02"))
    assert days(p, 10) == (d("2026-11-02"), d("2026-11-03"))


def test_a_not_before_earlier_than_the_start_date_changes_nothing():
    p = plan([task(1, desc="Estimate: 1 day. Not before: 2026-01-01")], START, NY)
    assert days(p, 1) == (START, START)


def test_a_blocker_later_than_the_floor_still_wins():
    tasks = [task(1, days=10), task(2, desc="Not before: 2026-10-07")]
    link(tasks, "blocking", 1, 2)
    p = plan(tasks, START, NY)
    assert days(p, 2) == (d("2026-10-15"), d("2026-10-15"))


def test_a_not_before_that_is_not_a_date_is_refused():
    with pytest.raises(ScheduleError, match="#1 Task 1 has a Not before date that is not a date"):
        plan([task(1, desc="Not before: 2026-13-45")], START, NY)


def test_a_done_task_with_a_not_before_is_ignored():
    p = plan([task(1, done=True, desc="Not before: 2026-13-45")], START, NY)
    assert p.floors == {}


# --- scope ------------------------------------------------------------------------------
def test_an_excluded_task_its_subtasks_and_its_successors_do_not_move():
    april = "2027-04-01T13:00:00Z"
    tasks = [
        task(7, labels=[66], start=april, end=april),
        task(8, days=1),
        task(9, days=1),
        task(20, days=1),
        task(21, days=1),
        task(30, days=3),
    ]
    link(tasks, "subtask", 7, 8)
    link(tasks, "blocking", 9, 8)
    link(tasks, "blocking", 8, 20)
    link(tasks, "blocking", 20, 21)
    p = plan(tasks, START, NY, exclude_label_id=66)
    assert p.excluded == {7: "labelled out of scope", 8: "subtask of #7", 20: "waits on #8", 21: "waits on #20"}
    assert set(p.dates) == {9, 30}
    assert {c.task_id for c in p.changes} == {9, 30}
    # The finish and the critical path are those of the work in scope.
    assert p.finish == d("2026-10-07")
    assert p.critical_path == [30]
    assert p.done == 0


def test_exclusion_names_each_task_once_and_skips_done_ones():
    # 20 is reached twice, through 7 and through its subtask 8. 9 is done.
    tasks = [task(7, labels=[66]), task(8, days=1), task(9, days=1, done=True), task(20, days=1)]
    link(tasks, "subtask", 7, 8)
    link(tasks, "subtask", 7, 9)
    link(tasks, "blocking", 7, 20)
    link(tasks, "blocking", 8, 20)
    p = plan(tasks, START, NY, exclude_label_id=66)
    assert p.excluded == {7: "labelled out of scope", 8: "subtask of #7", 20: "waits on #7"}


def test_excluded_tasks_lose_the_ready_and_critical_labels():
    tasks = [task(7, days=1, labels=[66, 5, 9]), task(8, days=1)]
    p = plan(tasks, START, NY, ready_label_id=5, critical_label_id=9, exclude_label_id=66)
    assert p.labels_removed == {7: [5, 9]}
    assert p.labels_added == {8: [5, 9]}


def test_a_parent_with_every_subtask_excluded_is_scheduled_on_its_own_estimate():
    tasks = [task(10, days=2), task(11, days=5, labels=[66])]
    link(tasks, "subtask", 10, 11)
    p = plan(tasks, START, NY, exclude_label_id=66)
    assert days(p, 10) == (d("2026-10-05"), d("2026-10-06"))
    assert p.excluded == {11: "labelled out of scope"}


def test_the_exclude_label_does_nothing_on_done_tasks_or_when_not_given():
    tasks = [task(1, days=1, done=True, labels=[66]), task(2, days=1, labels=[66])]
    assert plan(tasks, START, NY).excluded == {}
    assert plan(tasks, START, NY, exclude_label_id=66).excluded == {2: "labelled out of scope"}


def test_summary_lists_floors_and_out_of_scope_tasks():
    tasks = [task(1, title="Dig", desc="Not before: 2026-10-09"), task(2, title="AC", days=1, labels=[66])]
    text = summarise(plan(tasks, START, NY, exclude_label_id=66), {}, dry_run=True)
    assert "Held by Not before (1):\n- #1 Dig: not before 2026-10-09, starts 2026-10-09" in text
    assert "Not moved, out of scope (1):\n- #2 AC: labelled out of scope" in text
    assert "Unchanged: 0 open, 0 done left as they were." in text


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


# --- complete_task and reopen_task ------------------------------------------------------
def freeze(monkeypatch, day):
    """Make today `day` in New York, for the tools that default to today."""
    noon = datetime.fromisoformat(f"{day}T12:00:00-04:00")

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return noon.astimezone(tz)

    monkeypatch.setattr(schedule, "datetime", Frozen)


def at(day, hour):
    """A stored date: `hour` o'clock in New York on `day`, as Vikunja returns it in UTC."""
    local = datetime.fromisoformat(f"{day}T{hour:02d}:00:00-04:00")
    return local.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


def on(day, last=None):
    """Stored dates for a task planned across `day` to `last`."""
    return {"start": at(day, 9), "end": at(last or day, 17)}


def served(api, tasks, task_id, *writes):
    """Answer the task read, the project listing, then any writes."""
    target = next(t for t in tasks if t["id"] == task_id)
    api.returns_in_order(
        httpx.Response(200, json={**target, "project_id": 3}),
        httpx.Response(200, json={"items": tasks, "total_pages": 1}),
        *(writes or [httpx.Response(200, json={"ok": True})]),
    )


def calls(api):
    return [(r.method, r.url.path.removeprefix("/api/v2")) for r in api.requests]


V2 = pytest.mark.parametrize("api_version", [2])


@V2
def test_completing_a_task_makes_its_successor_ready(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, title="Dig", days=1, **on("2026-10-05")), task(2, title="Fill", days=1, **on("2026-10-06"))]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=True))
    assert text.splitlines()[:2] == ["Dry run. Nothing was written.", "#1 Dig — done, 2026-10-05 → 2026-10-05"]
    assert "Became ready (1):\n- #2 Fill" in text
    assert "- #2 Fill: 2026-10-06 → 2026-10-05" in text
    assert "Project finish: 2026-10-05, 1 day earlier than 2026-10-06." in text
    assert calls(api) == [("GET", "/tasks/1"), ("GET", "/projects/3/tasks")]


@V2
def test_a_successor_with_another_open_blocker_does_not_become_ready(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1), task(3, days=2), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    link(tasks, "blocking", 3, 2)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=True))
    assert "Became ready" not in text


@V2
def test_finishing_early_moves_the_finish_earlier(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=3, **on("2026-10-05", "2026-10-07")), task(2, days=1, **on("2026-10-08"))]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=True))
    assert "Project finish: 2026-10-05, 3 days earlier than 2026-10-08." in text


@V2
def test_finishing_late_moves_the_finish_later(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-08")
    tasks = [task(1, days=1, **on("2026-10-05")), task(2, days=1, **on("2026-10-06"))]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=True))
    assert "#1 Task 1 — done, 2026-10-05 → 2026-10-08" in text
    assert "Project finish: 2026-10-08, 2 days later than 2026-10-06." in text


@V2
def test_completing_the_last_subtask_leaves_the_parent_open_and_ready(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(10, title="Parent"), task(11, days=1, done=True), task(12, days=1)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    served(api, tasks, 12)
    text = run(schedule.complete_task(12, ready_label_id=5))
    assert "Became ready (1):\n- #10 Parent" in text
    assert "Every subtask now done, parent left open for you to close:\n- #10 Parent" in text
    # The parent is re-dated as an ordinary task now, and labelled ready. It is never
    # marked done.
    parent_writes = [json.loads(r.content) for r in api.requests if r.url.path.endswith("/tasks/10")]
    assert parent_writes and all("done" not in w for w in parent_writes)
    assert ("POST", "/tasks/10/labels") in calls(api)


@V2
def test_a_parent_with_another_open_subtask_is_not_mentioned(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(10), task(11, days=1), task(12, days=1), task(20, done=True)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    link(tasks, "subtask", 20, 12)
    served(api, tasks, 12)
    with pytest.raises(ScheduleError, match="two parents"):
        run(schedule.complete_task(12, dry_run=True))
    tasks = [task(10), task(11, days=1), task(12, days=1)]
    link(tasks, "subtask", 10, 11)
    link(tasks, "subtask", 10, 12)
    served(api, tasks, 12)
    assert "parent left open" not in run(schedule.complete_task(12, dry_run=True))


@V2
def test_a_done_parent_is_not_mentioned(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(10, done=True), task(12, days=1)]
    link(tasks, "subtask", 10, 12)
    served(api, tasks, 12)
    assert "parent left open" not in run(schedule.complete_task(12, dry_run=True))


@pytest.mark.parametrize("dry_run", [False, True])
@V2
def test_completing_a_task_that_is_already_done_changes_nothing(api, run, api_version, dry_run):
    tasks = [task(1, title="Dig", done=True)]
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=dry_run))
    assert text.endswith("#1 Dig is already done. Nothing was changed.")
    assert text.startswith("Dry run.") == dry_run
    assert calls(api) == [("GET", "/tasks/1")]


@V2
def test_complete_writes_the_task_first_then_dates_then_labels(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1, labels=[5, 9], **on("2026-10-03")), task(2, days=1, **on("2026-10-09"))]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    run(schedule.complete_task(1, ready_label_id=5, critical_label_id=9))
    assert calls(api) == [
        ("GET", "/tasks/1"),
        ("GET", "/projects/3/tasks"),
        ("PATCH", "/tasks/1"),
        ("PATCH", "/tasks/2"),
        ("POST", "/tasks/2/labels"),
        ("POST", "/tasks/2/labels"),
        ("DELETE", "/tasks/1/labels/5"),
        ("DELETE", "/tasks/1/labels/9"),
    ]
    # The stored start is kept, normalised to 09:00.
    assert json.loads(api.requests[2].content) == {
        "done": True,
        "start_date": "2026-10-03T09:00:00-04:00",
        "end_date": "2026-10-05T17:00:00-04:00",
    }


@V2
def test_complete_takes_an_explicit_start_and_completion_date(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    served(api, [task(1, days=1)], 1)
    run(schedule.complete_task(1, completed_date="2026-11-02", actual_start_date="2026-10-30"))
    assert json.loads(api.requests[2].content) == {
        "done": True,
        "start_date": "2026-10-30T09:00:00-04:00",
        "end_date": "2026-11-02T17:00:00-05:00",
    }


@pytest.mark.parametrize("stored", [{}, on("2026-10-09")])
@V2
def test_complete_starts_on_the_completion_date_without_an_earlier_start(api, run, monkeypatch, api_version, stored):
    freeze(monkeypatch, "2026-10-05")
    served(api, [task(1, days=1, **stored)], 1)
    run(schedule.complete_task(1))
    assert json.loads(api.requests[2].content)["start_date"] == "2026-10-05T09:00:00-04:00"


@V2
def test_complete_refuses_a_start_after_the_completion(api, run, api_version):
    served(api, [task(1, days=1)], 1)
    with pytest.raises(ValueError, match="is after completed_date"):
        run(schedule.complete_task(1, completed_date="2026-10-05", actual_start_date="2026-10-06"))
    assert calls(api) == [("GET", "/tasks/1")]


@V2
def test_complete_without_reschedule_keeps_dates_but_moves_labels(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1, **on("2026-10-05")), task(2, days=1, **on("2026-10-09"))]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, reschedule=False, ready_label_id=5))
    assert ("PATCH", "/tasks/2") not in calls(api)
    assert ("POST", "/tasks/2/labels") in calls(api)
    assert "Date changes (0):\n- none" in text
    assert "Project finish: 2026-10-09, unchanged." in text


@V2
def test_complete_reports_the_critical_path_moving(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    # Before: 1 (3 days) then 3 is critical, 2 has slack. After 1 is done, 2 then 3 is.
    tasks = [task(1, days=3, labels=[9]), task(2, days=1), task(3, days=1, labels=[9])]
    link(tasks, "blocking", 1, 3)
    link(tasks, "blocking", 2, 3)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, critical_label_id=9, dry_run=True))
    assert "Joined the critical path (1):\n- #2 Task 2" in text
    assert "Left the critical path" not in text


@V2
def test_complete_without_label_ids_compares_computed_sets(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1), task(2, days=1), task(3, days=1)]
    link(tasks, "blocking", 1, 3)
    link(tasks, "blocking", 2, 3)
    served(api, tasks, 2)
    text = run(schedule.complete_task(2, dry_run=True))
    assert "Became ready" not in text
    assert "Left the critical path" not in text


@V2
def test_complete_keeps_out_of_scope_work_out(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1), task(7, title="AC", days=1, labels=[66], **on("2027-04-01"))]
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, exclude_label_id=66, dry_run=True))
    assert "Not moved, out of scope (1):\n- #7 AC: labelled out of scope" in text
    assert "Project finish: no open tasks remain in scope." in text


@V2
def test_complete_reports_a_cycle_it_breaks_without_a_before_plan(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    link(tasks, "blocking", 2, 1)
    served(api, tasks, 1)
    text = run(schedule.complete_task(1, dry_run=True))
    assert "Project finish: 2026-10-05, unchanged." in text


@V2
def test_a_failed_completion_points_at_reschedule_project(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, days=1), task(2, days=1)]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1, httpx.Response(200, json={"id": 1}), httpx.Response(500, json={"message": "boom"}))
    with pytest.raises(RuntimeError) as err:
        run(schedule.complete_task(1))
    assert "Already written: done on #1." in str(err.value)
    assert "Run reschedule_project to finish the rest." in str(err.value)


def test_complete_refuses_an_unreadable_task(api, run):
    api.returns_raw(204)
    with pytest.raises(RuntimeError, match="did not return task 1"):
        run(schedule.complete_task(1))


@V2
def test_reopen_restores_the_labels(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    tasks = [task(1, title="Dig", days=1, done=True, **on("2026-10-02")), task(2, title="Fill", days=1, labels=[5])]
    link(tasks, "blocking", 1, 2)
    served(api, tasks, 1)
    text = run(schedule.reopen_task(1, ready_label_id=5))
    assert calls(api)[2:] == [
        ("PATCH", "/tasks/1"),
        ("PATCH", "/tasks/1"),
        ("PATCH", "/tasks/2"),
        ("POST", "/tasks/1/labels"),
        ("DELETE", "/tasks/2/labels/5"),
    ]
    assert json.loads(api.requests[2].content) == {"done": False}
    assert text.splitlines()[1] == "#1 Dig — reopened, 2026-10-05"
    assert "Lost ready (1):\n- #2 Fill" in text
    assert "parent left open" not in text


@V2
def test_reopen_without_reschedule_leaves_dates(api, run, monkeypatch, api_version):
    freeze(monkeypatch, "2026-10-05")
    served(api, [task(1, days=1, done=True, **on("2026-10-02"))], 1)
    text = run(schedule.reopen_task(1, reschedule=False, dry_run=True))
    assert text.splitlines()[1] == "#1 Task 1 — reopened, dates unchanged"
    assert calls(api) == [("GET", "/tasks/1"), ("GET", "/projects/3/tasks")]


@V2
def test_reopening_an_open_task_changes_nothing(api, run, api_version):
    served(api, [task(1, title="Dig", days=1)], 1)
    text = run(schedule.reopen_task(1))
    assert text == "#1 Dig is already open. Nothing was changed."
    assert calls(api) == [("GET", "/tasks/1")]
