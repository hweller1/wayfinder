"""Prompt nudges appended to scout's product prompt (no scout imports, so CPU tests can load it).

`teacher` is the convergence nudge every earlier eval and rollout used (run.TEACHER_NUDGE). `f1` adds one line for GRPO run f1: the F1 reward scores
the first 12 reported files and the all-gold bonus needs the gold files in the first 8, so the model is told the order matters. THIS CHANGES THE
HARNESS: the f1 run's rollouts and every eval of its checkpoints use it (WF_EVAL_NUDGE=f1, cfg["nudge"]="f1"); the v3 and A1 reference rows were
collected without it. Scout's prompt file is not modified; the exact text is recorded in every row's `nudge`.
"""

TEACHER_NUDGE = "Report as soon as the evidence covers the brief; do not exhaust the turn budget."
ORDER_LINE = "The files listed first in your report count the most, so list the most relevant files first."
NUDGES = {"teacher": TEACHER_NUDGE, "f1": TEACHER_NUDGE + "\n\n" + ORDER_LINE}
