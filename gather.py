"""Community contributions → one raw dataset. Step 1 of building a community
LeRobot dataset; the conversion form below it is step 2.

A community dataset holds no recordings. Each accepted episode is one small JSON
under ``contrib/<contributor>/`` pointing at a recording that lives in the
CONTRIBUTOR's own raw dataset. Copying those into a single raw dataset is what
turns many contributors' uploads into something the converter can read — and it
belongs HERE rather than on a maintainer's laptop, because both ends are on the
Hub: run from home, every episode would come down a domestic link and go straight
back up again.

Deliberately NOT folded into the run generator in controller.py. This has no SLAM,
no episode review, no pause: it is a file copy, and giving it its own small panel
keeps both flows readable. It does reuse controller._preflight, which already
answers exactly the two questions asked here (can I read the source, can I create
or write the target).
"""

import queue
import threading

import gradio as gr
from huggingface_hub import HfApi

from controller import _preflight
from grabette_postprocess.community import ALREADY_GATHERED, skipped_lines
from grabette_postprocess.community import gather as gather_contributions

# Whether the Gather button is usable, and what it says while it is not.
# Same arrow as the conversion's Run button: both start work, so they should not
# look like different kinds of control.
_IDLE = "▶  Gather into a raw dataset"
_BUSY = "Gathering…"


def _render(title, checks=(), counts=(), detail=(), footer="") -> str:
    """One message, in the shape the conversion's own summary uses.

    Everything the gathering has to say goes here — the access checks, the
    counts, what was left out and why, the result — because a log box beside a
    summary made the reader join two halves themselves, and the interesting half
    was never the one they were looking at.

    checks: (ok, text) pairs, rendered ✅ or ❌.
    counts: (number, text) pairs, one per LINE: the numbers are the answer to
        "did it do anything", and a sentence hides them.
    """
    lines = [f"### {title}"]
    lines += [("✅ " if ok else "❌ ") + text for ok, text in checks]
    if counts:
        lines.append("")
        lines += [f"**{n}** {text}" for n, text in counts]
    if detail:
        lines.append("")
        lines += list(detail)
    if footer:
        lines.append("")
        lines.append(footer)
    return "\n".join(lines)


def esc_md(text) -> str:
    """Keep a repo id or a Hub error from being read as Markdown."""
    out = str(text)
    for ch in ("\\", "`", "*", "_", "[", "]"):
        out = out.replace(ch, "\\" + ch)
    return out


def _counts(plan_seen, community_repo, *, done, plan=None):
    """The numbers, one per line: contributions seen, gathered, left out for a
    reason worth reading, and already gathered.

    Split that way because the three mean different things. "8 already gathered"
    is reassurance that a re-run is doing nothing wrong; "1 left out" is the one
    that needs looking at; folding them into a single "9 skipped" hid the second
    behind the first.

    Reads the plan the library hands over BEFORE copying (plan_seen), so the
    counts are on screen while the copy runs, not only after it.
    """
    p = plan or (plan_seen[0] if plan_seen else None)
    if p is None:
        return []
    already = sum(1 for _c, why in p.skipped if why == ALREADY_GATHERED)
    others = len(p.skipped) - already
    out = [(len(p.gather) + len(p.skipped), f"contribution(s) in `{community_repo}`"),
           (len(p.gather), "gathered" if done else "to gather")]
    if others:
        out.append((others, "left out — see below"))
    if already:
        out.append((already, "already gathered, left as they are"))
    return out


def _view(message="", *, busy, source=None, proceed=False, chained=False):
    """The panel's four outputs: the message, the button, the conversion form's
    source field, and whether the conversion may proceed.

    `proceed` defaults to False on EVERY yield, so a refusal, a dry run or a
    failure can never leave a stale True behind for the .then() to act on.

    `chained` says the gathering is Run's first half rather than the button's own
    click. The button then stays disabled throughout, because with "…and convert"
    ticked it is Run that owns the action.
    """
    return (
        message,
        gr.update(value=_BUSY if busy else _IDLE,
                  interactive=not busy and not chained),
        gr.skip() if source is None else gr.update(value=source),
        proceed,
    )


def _passthrough():
    """Write nothing, and let the conversion proceed: Run with "…and convert"
    unticked must behave exactly as it always has, gather panel untouched."""
    return (gr.skip(), gr.skip(), gr.skip(), True)


def _source_only_check(api, repo):
    """Can we read the community dataset? The whole pre-flight for a dry run with
    no target. Deliberately not _preflight: that also demands write access to a
    target, which a preview has no business requiring."""
    try:
        api.repo_info(repo, repo_type="dataset")
        return None
    except Exception as e:  # noqa: BLE001 — a message for a human, not a traceback
        return (f"Error on the community dataset '{repo}':\n"
                f"Either it doesn't exist, or it's private and your account cannot "
                f"see it. Check the spelling; it should look like "
                f"'namespace/dataset-name'. ({e.__class__.__name__})")


# ── the panel's interactive states ───────────────────────────────────────────
# Three small handlers rather than one: each writes only the components it must,
# so none of them ever writes the field being typed in or the box being clicked
# (which would fight the user for focus).

def gather_ready(community, raw, dry, chain):
    """Is the Gather button usable? Only with what it needs filled in, and never
    while "…and convert" is ticked — Run owns the action then."""
    filled = bool((community or "").strip()) and (bool(dry) or bool((raw or "").strip()))
    return gr.update(interactive=filled and not bool(chain))


def dry_toggled(community, raw, dry, chain):
    """A dry run locks the conversion, and nothing else.

    It is tempting to grey out everything a preview "does not need", and wrong:
      * the raw dataset IS used — it is what the preview compares against to say
        which episodes are new rather than already gathered. Locking it made the
        more useful kind of dry run impossible to set up;
      * the device filter IS used — it decides what the preview reports as left
        out.
    Only converting is meaningless with nothing copied, so only that is locked.
    """
    return (gr.update(interactive=not bool(dry)),
            gather_ready(community, raw, dry, chain))


def chain_toggled(community, raw, dry, chain):
    """With "…and convert" ticked, Run does both steps: Dry run is meaningless
    (there would be nothing to convert) and the Gather button is superseded."""
    return (gr.update(interactive=not bool(chain)),
            gather_ready(community, raw, dry, chain),
            mirror_source(chain, raw))


def mirror_source(chain, raw):
    """Run reads Source dataset, so with "…and convert" ticked it must already say
    the raw dataset the gathering will write. Mirrored live rather than only at
    the end, so what Run will use is visible before pressing it."""
    return gr.update(value=(raw or "").strip()) if chain else gr.skip()


def gather_before_run(chain, community_repo, raw_repo, roles,
                      conv_target="", conv_task="",
                      oauth_token: gr.OAuthToken | None = None):
    """Run's first half: gather when "…and convert" is ticked, otherwise nothing.

    This is what makes Run do both steps. With the box unticked it writes nothing
    at all and lets the conversion run exactly as it always has.
    """
    if not bool(chain):
        yield _passthrough()
        return
    yield from run_gather(community_repo, raw_repo, roles, False, True,
                          conv_target, conv_task, oauth_token=oauth_token)


def run_gather(community_repo, raw_repo, roles, dry_run, chain=False,
               conv_target="", conv_task="",
               oauth_token: gr.OAuthToken | None = None):
    """Streaming generator behind the Gather button.

    Yields (log, summary, button, source, chain). `source` is the point of doing
    this in the Space at all: on success the conversion form's Source dataset is
    filled in with the raw dataset just written, so step 2 is one click away.
    `chain` tells the conversion chained after this one whether to run.

    conv_target / conv_task are the conversion form's fields, read here ONLY to
    validate them: "and convert" with an empty target would copy every episode
    and then fail on a missing field, which is the worst possible ordering.
    """
    if oauth_token is None:
        raise gr.Error("Sign in with your Hugging Face account first.")
    community_repo = (community_repo or "").strip()
    raw_repo = (raw_repo or "").strip()
    if not community_repo:
        raise gr.Error("Fill in the community dataset.")
    # A dry run writes nothing, so it does not need a target. Reported without one
    # it simply cannot say what is ALREADY gathered — see the log line for that.
    if not (raw_repo or dry_run):
        raise gr.Error("Fill in the raw dataset to write.")
    if raw_repo and community_repo == raw_repo:
        # Gathering a repo into itself would write episode folders into the
        # community dataset, which is precisely what this design keeps out of it.
        raise gr.Error("The raw dataset must be a different repo from the community one.")
    chain = bool(chain) and not dry_run  # a dry run copies nothing to convert
    if chain and not ((conv_target or "").strip() and (conv_task or "").strip()):
        raise gr.Error("To gather and convert in one go, fill in Target dataset and "
                       "Task description in the form below first.")
    token = oauth_token.token  # local only — never os.environ, see controller._run
    scopes = frozenset((getattr(oauth_token, "scope", "") or "").split())

    api = HfApi(token=token)
    yield _view(_render("Gathering community contributions"), busy=True, chained=chain)

    # ---- access ------------------------------------------------------------
    # The target may well already exist (a second gathering adds to it), so only
    # the error matters — unlike a conversion, where an existing target means
    # something. With no target at all (a dry run), only the source is checked:
    # demanding write access to a repo nobody will write to would refuse a
    # perfectly good preview.
    if raw_repo:
        _exists, _writable, err = _preflight(api, community_repo, raw_repo, scopes)
    else:
        err = _source_only_check(api, community_repo)
    if err:
        yield _view(_render("Cannot gather", checks=[(False, err.replace("\n", " "))]),
                    busy=False, chained=chain)
        return
    checks = [(True, f"`{community_repo}` readable")]
    if raw_repo:
        checks.append((True, f"write access to `{raw_repo.split('/')[0]}`"))
    yield _view(_render("Gathering community contributions", checks=checks),
                busy=True, chained=chain)

    # ---- the work ----------------------------------------------------------
    q: "queue.Queue[str]" = queue.Queue()
    result: dict = {}
    plan_seen: list = []

    def work():
        try:
            result["plan"] = gather_contributions(
                community_repo, raw_repo, token=token,
                require_roles=list(roles or ()) or None,
                dry_run=bool(dry_run), log=q.put, on_plan=plan_seen.append)
        except Exception as e:  # noqa: BLE001 — surfaced in the panel, never a traceback
            result["error"] = str(e)
        finally:
            q.put(None)  # sentinel: the worker is done

    t = threading.Thread(target=work, daemon=True)
    t.start()
    last = ""
    while True:
        line = q.get()
        if line is None:
            break
        # The library's own log lines are progress, not the report: only the
        # newest is shown, under the counts, so the message stays one screenful.
        last = line.strip()
        yield _view(_render("Gathering community contributions", checks=checks,
                            counts=_counts(plan_seen, community_repo, done=False),
                            footer=f"_{esc_md(last)}_" if last else ""),
                    busy=True, chained=chain)

    if "error" in result:
        yield _view(_render("Gathering failed", checks=checks + [(False, esc_md(result["error"]))]),
                    busy=False, chained=chain)
        return

    plan = result["plan"]
    others = [(c, why) for c, why in plan.skipped if why != ALREADY_GATHERED]
    detail = [l.strip() for l in skipped_lines(others)] if others else []
    n = len(plan.gather)
    url = f"https://huggingface.co/datasets/{raw_repo}" if raw_repo else ""

    if dry_run:
        footer = (f"Nothing was copied. Untick **Dry run** to gather them."
                  if n else "Nothing new to copy.")
        if not raw_repo:
            footer += ("\n\nWith no raw dataset named, this cannot tell what is already "
                       "gathered — fill one in to see only what is new.")
        yield _view(_render("Dry run", checks=checks,
                            counts=_counts(plan_seen, community_repo, done=False, plan=plan),
                            detail=detail, footer=footer),
                    busy=False, chained=chain)
        return

    if not n:
        # Not a failure: everything merged so far is already there.
        footer = f"Nothing new to copy — [{raw_repo}]({url}) is up to date."
        yield _view(_render("Already up to date", checks=checks,
                            counts=_counts(plan_seen, community_repo, done=True, plan=plan),
                            detail=detail, footer=footer),
                    busy=False, chained=chain,
                    source=raw_repo if chain else None, proceed=chain)
        return

    footer = f"**Raw dataset:** 👉 [{raw_repo}]({url})"
    footer += (f"\n\n**Converting it now** into `{conv_target.strip()}` — the run "
               f"panel takes over from here."
               if chain else
               f"\n\nThis is a **raw** dataset, not a LeRobot one. Put it in "
               f"**Source dataset** below, with a target and a task description, "
               f"then **Run**.")
    yield _view(_render(f"{n} episode(s) gathered", checks=checks,
                        counts=_counts(plan_seen, community_repo, done=True, plan=plan),
                        detail=detail, footer=footer),
                busy=False, chained=chain,
                source=raw_repo if chain else None, proceed=chain)
