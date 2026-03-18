#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import textwrap
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ORCHESTRATOR_DIR = Path(".orchestrator")
RUNS_DIR = ORCHESTRATOR_DIR / "runs"
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_IDENTICAL_FAILURES = 2


@dataclass
class IterationResult:
    worker_exit_code: int
    worker_text: str
    worker_thread_id: str | None
    verification_results: list[dict[str, Any]]
    status_lines: list[str]
    diff_stat: str
    success_file_results: list[dict[str, Any]]
    failure_signature: str


class CommandTimeoutError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / "iterations").mkdir(parents=True, exist_ok=True)


def read_text(path: Path | None) -> str:
    if not path:
        return ""
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_job_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config is not supported by this Python build.")
        return tomllib.loads(text)
    raise ValueError("Job files must be .json or .toml")


def latest_run_dir() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_dir(job_id: str | None) -> Path:
    if job_id:
        run_dir = RUNS_DIR / job_id
    else:
        run_dir = latest_run_dir()
        if run_dir is None:
            raise SystemExit("No orchestration runs found.")
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    return run_dir


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
    )


def run_shell_command(command: str, *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            executable=os.environ.get("SHELL", "/bin/zsh"),
            timeout=max(1, timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "exit_code": 124,
            "stdout": stdout,
            "stderr": (stderr + f"\nTimed out after {timeout_seconds}s.").strip(),
            "timed_out": True,
        }
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": False,
    }


def get_git_status_lines(cwd: Path) -> list[str]:
    completed = run_command(["git", "status", "--short"], cwd=cwd)
    if completed.returncode != 0:
        return [f"git status failed: {completed.stderr.strip()}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def get_git_diff_stat(cwd: Path) -> str:
    completed = run_command(["git", "diff", "--stat"], cwd=cwd)
    if completed.returncode != 0:
        return f"git diff --stat failed: {completed.stderr.strip()}"
    return completed.stdout.strip()


def summarize_git_state(status_lines: list[str], diff_stat: str) -> str:
    status_preview = "\n".join(status_lines[:20]) if status_lines else "No changed tracked files."
    diff_preview = diff_stat or "No tracked diff stats."
    return f"git status --short:\n{status_preview}\n\ngit diff --stat:\n{diff_preview}"


def normalize_failure_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = []
    for line in lines[:8]:
        compact = " ".join(line.split())
        normalized.append(compact[:240])
    return normalized


def build_failure_signature(
    verification_results: list[dict[str, Any]],
    diff_stat: str,
    worker_text: str,
) -> str:
    parts: list[str] = []
    failures = [result for result in verification_results if result["exit_code"] != 0]
    for result in failures:
        parts.append(f"verify:{result['command']}")
        parts.extend(normalize_failure_lines(result["stderr"] or result["stdout"]))
    if not failures:
        parts.append("verify:ok")
    parts.append(f"diff:{' '.join(diff_stat.split())[:240] or 'none'}")
    parts.extend(f"worker:{line}" for line in normalize_failure_lines(worker_text))
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def count_consecutive_identical_failures(state: dict[str, Any], signature: str) -> int:
    count = 0
    for iteration in reversed(state["iterations"]):
        if iteration.get("failure_signature") == signature and iteration.get("decision") != "success":
            count += 1
        else:
            break
    return count


def build_success_file_results(patterns: list[str], cwd: Path) -> list[dict[str, Any]]:
    results = []
    for pattern in patterns:
        matches = sorted(str(path.relative_to(cwd)) for path in cwd.glob(pattern))
        results.append({"pattern": pattern, "matches": matches, "ok": bool(matches)})
    return results


def verification_failed(
    verification_results: list[dict[str, Any]],
    success_file_results: list[dict[str, Any]],
) -> bool:
    if any(result["exit_code"] != 0 for result in verification_results):
        return True
    return any(not result["ok"] for result in success_file_results)


def build_verification_summary(
    verification_results: list[dict[str, Any]],
    success_file_results: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    if verification_results:
        for result in verification_results:
            lines.append(f"$ {result['command']}")
            lines.append(f"exit_code={result['exit_code']}")
            stdout = (result["stdout"] or "").strip()
            stderr = (result["stderr"] or "").strip()
            if stdout:
                lines.append("stdout:")
                lines.extend(stdout.splitlines()[:30])
            if stderr:
                lines.append("stderr:")
                lines.extend(stderr.splitlines()[:30])
            lines.append("")
    else:
        lines.append("No verification commands configured.")
    if success_file_results:
        lines.append("Success file checks:")
        for result in success_file_results:
            lines.append(
                f"{result['pattern']}: {'matched' if result['ok'] else 'missing'}"
                + (f" -> {', '.join(result['matches'])}" if result["matches"] else "")
            )
    return "\n".join(lines).strip()


def load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            events.append({"type": "raw", "text": stripped})
    return events


def parse_codex_output(jsonl_path: Path, last_message_path: Path) -> tuple[str | None, str]:
    thread_id = None
    for event in load_jsonl_events(jsonl_path):
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
    message = last_message_path.read_text(encoding="utf-8").strip() if last_message_path.exists() else ""
    if not message:
        for event in reversed(load_jsonl_events(jsonl_path)):
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                message = item["text"].strip()
                break
    return thread_id, message


def build_initial_prompt(
    *,
    goal: str,
    context_text: str,
    verify_commands: list[str],
    success_patterns: list[str],
) -> str:
    verification_lines = "\n".join(f"- `{cmd}`" for cmd in verify_commands) or "- None configured"
    success_lines = "\n".join(f"- `{pattern}`" for pattern in success_patterns) or "- None configured"
    prompt = f"""
    You are the worker Codex session for this repository.

    Goal:
    {goal}

    Extra context:
    {context_text or "No extra context provided."}

    Requirements:
    - Make the repository satisfy the goal with real code changes when needed.
    - Explore the codebase before changing code.
    - Run relevant checks after edits when practical.
    - In your final message, summarize what changed, what you verified, and any remaining risk.
    - Do not stop at analysis if implementation is needed.
    - If the goal involves a failed workflow, fix the underlying cause and preserve successful existing behavior.
    - Prefer durable extractor, retry, state, and output fixes over one-off hacks.

    The supervisor will verify your work using:
    {verification_lines}

    The supervisor will also check for expected files:
    {success_lines}
    """
    return textwrap.dedent(prompt).strip() + "\n"


def build_retry_prompt(
    *,
    goal: str,
    prior_worker_text: str,
    verification_summary: str,
    git_summary: str,
    consecutive_failures: int,
) -> str:
    prompt = f"""
    Continue working on this repository goal:
    {goal}

    Your previous attempt did not satisfy the supervisor checks.

    Previous final message:
    {prior_worker_text or "No worker summary captured."}

    Verification results:
    {verification_summary}

    Repository state after your attempt:
    {git_summary}

    Retry guidance:
    - Fix the failing behavior instead of repeating the same approach.
    - Preserve good changes that should remain.
    - If a check failure is unrelated noise, explain why briefly and still finish the main fix.
    - Avoid returning success unless the repo is actually in a good state.
    - This is consecutive failed attempt number {consecutive_failures + 1} for the same failure signature.
    - If the same failure keeps recurring, change the design or add missing diagnostics instead of only tweaking prompt wording.

    In your final message, include:
    - changed files
    - checks you ran
    - whether you believe the supervisor verification will now pass
    """
    return textwrap.dedent(prompt).strip() + "\n"


def load_context_text(args: argparse.Namespace, job_config: dict[str, Any]) -> str:
    context_path = args.context_file or job_config.get("context_file")
    if not context_path:
        return ""
    return Path(context_path).read_text(encoding="utf-8").strip()


def resolve_verify_commands(args: argparse.Namespace, job_config: dict[str, Any]) -> list[str]:
    commands = list(job_config.get("verify", []))
    if args.verify:
        commands = list(args.verify)
    return commands


def resolve_success_patterns(args: argparse.Namespace, job_config: dict[str, Any]) -> list[str]:
    patterns = list(job_config.get("success_file_patterns", []))
    if args.success_file_pattern:
        patterns = list(args.success_file_pattern)
    return patterns


def build_worker_command(
    *,
    cwd: Path,
    last_message_path: Path,
    worker_thread_id: str | None,
    worker_model: str | None,
) -> list[str]:
    if worker_thread_id:
        base = [
            "codex",
            "exec",
            "resume",
            "--json",
            "--full-auto",
            "--skip-git-repo-check",
            "-C",
            str(cwd),
            "-o",
            str(last_message_path),
        ]
    else:
        base = [
            "codex",
            "exec",
            "--json",
            "--full-auto",
            "--skip-git-repo-check",
            "-C",
            str(cwd),
            "-o",
            str(last_message_path),
        ]
    if worker_model:
        base.extend(["-m", worker_model])
    if worker_thread_id:
        base.append(worker_thread_id)
    base.append("-")
    return base


def run_worker_iteration(
    *,
    cwd: Path,
    run_dir: Path,
    state: dict[str, Any],
    prompt_text: str,
    worker_model: str | None,
) -> IterationResult:
    iteration_number = state["iteration"] + 1
    iteration_dir = run_dir / "iterations" / f"{iteration_number:03d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = iteration_dir / "prompt.txt"
    jsonl_path = iteration_dir / "worker.jsonl"
    last_message_path = iteration_dir / "last_message.txt"
    verification_path = iteration_dir / "verification.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    worker_cmd = build_worker_command(
        cwd=cwd,
        last_message_path=last_message_path,
        worker_thread_id=state.get("worker_thread_id"),
        worker_model=worker_model,
    )
    completed = run_command(worker_cmd, cwd=cwd, stdin_text=prompt_text)
    jsonl_path.write_text(completed.stdout, encoding="utf-8")
    thread_id, worker_text = parse_codex_output(jsonl_path, last_message_path)

    verification_results = [
        run_shell_command(command, cwd=cwd, timeout_seconds=state["verify_timeout_seconds"])
        for command in state["verify_commands"]
    ]
    success_file_results = build_success_file_results(state["success_file_patterns"], cwd)
    status_lines = get_git_status_lines(cwd)
    diff_stat = get_git_diff_stat(cwd)
    failure_signature = build_failure_signature(verification_results, diff_stat, worker_text)
    write_json(
        verification_path,
        {
            "verification_results": verification_results,
            "success_file_results": success_file_results,
            "status_lines": status_lines,
            "diff_stat": diff_stat,
            "failure_signature": failure_signature,
        },
    )
    return IterationResult(
        worker_exit_code=completed.returncode,
        worker_text=worker_text,
        worker_thread_id=thread_id,
        verification_results=verification_results,
        status_lines=status_lines,
        diff_stat=diff_stat,
        success_file_results=success_file_results,
        failure_signature=failure_signature,
    )


def decide_iteration(
    *,
    result: IterationResult,
    state: dict[str, Any],
    max_consecutive_identical_failures: int,
) -> tuple[str, str]:
    if result.worker_exit_code != 0:
        return "failed", f"worker exited with code {result.worker_exit_code}"
    if not verification_failed(result.verification_results, result.success_file_results):
        return "success", "verification passed"
    consecutive = count_consecutive_identical_failures(state, result.failure_signature)
    if consecutive >= max_consecutive_identical_failures:
        return "stuck", f"failure signature repeated {consecutive + 1} times"
    return "retry", "verification failed"


def persist_iteration(
    *,
    state: dict[str, Any],
    result: IterationResult,
    decision: str,
    decision_reason: str,
) -> None:
    state["iteration"] += 1
    if result.worker_thread_id:
        state["worker_thread_id"] = result.worker_thread_id
    record = {
        "iteration": state["iteration"],
        "worker_exit_code": result.worker_exit_code,
        "worker_text": result.worker_text,
        "worker_thread_id": result.worker_thread_id,
        "verification_results": result.verification_results,
        "success_file_results": result.success_file_results,
        "status_lines": result.status_lines,
        "diff_stat": result.diff_stat,
        "failure_signature": result.failure_signature,
        "decision": decision,
        "decision_reason": decision_reason,
        "completed_at": utc_now(),
    }
    state["iterations"].append(record)
    state["last_failure_signature"] = result.failure_signature
    state["status"] = decision


def print_iteration_summary(state: dict[str, Any]) -> None:
    latest = state["iterations"][-1]
    print(f"[iteration {latest['iteration']}] decision={latest['decision']} reason={latest['decision_reason']}")
    worker_text = latest["worker_text"].strip()
    if worker_text:
        preview = worker_text.splitlines()[0][:200]
        print(f"worker: {preview}")
    for result in latest["verification_results"]:
        print(f"verify: {result['command']} -> exit {result['exit_code']}")


def initialize_state(args: argparse.Namespace, cwd: Path, job_config: dict[str, Any]) -> dict[str, Any]:
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    goal = args.goal or job_config.get("goal")
    if not goal:
        raise SystemExit("A goal is required via --goal or job config.")
    state = {
        "job_id": job_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "cwd": str(cwd),
        "goal": goal,
        "worker_thread_id": None,
        "status": "running",
        "iteration": 0,
        "last_failure_signature": None,
        "verify_commands": resolve_verify_commands(args, job_config),
        "success_file_patterns": resolve_success_patterns(args, job_config),
        "context_text": load_context_text(args, job_config),
        "max_iterations": args.max_iterations or job_config.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        "max_consecutive_identical_failures": args.max_consecutive_identical_failures
        or job_config.get("max_consecutive_identical_failures", DEFAULT_MAX_IDENTICAL_FAILURES),
        "verify_timeout_seconds": args.verify_timeout_seconds or job_config.get("verify_timeout_seconds", 900),
        "worker_model": args.worker_model or job_config.get("worker", {}).get("model"),
        "iterations": [],
    }
    return state


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(run_dir / "state.json", state)


def run_loop(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    cwd = Path(state["cwd"])
    while state["iteration"] < state["max_iterations"]:
        if state["iteration"] == 0:
            prompt_text = build_initial_prompt(
                goal=state["goal"],
                context_text=state["context_text"],
                verify_commands=state["verify_commands"],
                success_patterns=state["success_file_patterns"],
            )
        else:
            latest = state["iterations"][-1]
            prompt_text = build_retry_prompt(
                goal=state["goal"],
                prior_worker_text=latest["worker_text"],
                verification_summary=build_verification_summary(
                    latest["verification_results"],
                    latest["success_file_results"],
                ),
                git_summary=summarize_git_state(latest["status_lines"], latest["diff_stat"]),
                consecutive_failures=count_consecutive_identical_failures(
                    state,
                    latest["failure_signature"],
                ),
            )
        result = run_worker_iteration(
            cwd=cwd,
            run_dir=run_dir,
            state=state,
            prompt_text=prompt_text,
            worker_model=state.get("worker_model"),
        )
        decision, reason = decide_iteration(
            result=result,
            state=state,
            max_consecutive_identical_failures=state["max_consecutive_identical_failures"],
        )
        persist_iteration(state=state, result=result, decision=decision, decision_reason=reason)
        save_state(run_dir, state)
        print_iteration_summary(state)
        if decision in {"success", "failed", "stuck"}:
            return state
    state["status"] = "failed"
    state["stop_reason"] = f"max iterations reached ({state['max_iterations']})"
    save_state(run_dir, state)
    return state


def cmd_run(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    job_config = load_job_file(Path(args.job_file)) if args.job_file else {}
    state = initialize_state(args, cwd, job_config)
    run_dir = RUNS_DIR / state["job_id"]
    ensure_run_dirs(run_dir)
    save_state(run_dir, state)
    final_state = run_loop(run_dir, state)
    print(f"run_id: {final_state['job_id']}")
    print(f"status: {final_state['status']}")
    return 0 if final_state["status"] == "success" else 1


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.job_id)
    state = load_json(run_dir / "state.json")
    if state["status"] == "success":
        print("Run already succeeded.")
        return 0
    if state["status"] in {"stuck", "failed"} and not args.force:
        print("Run is stopped. Use --force to continue from the last state.")
        return 1
    state["status"] = "running"
    save_state(run_dir, state)
    final_state = run_loop(run_dir, state)
    print(f"run_id: {final_state['job_id']}")
    print(f"status: {final_state['status']}")
    return 0 if final_state["status"] == "success" else 1


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.job_id)
    state = load_json(run_dir / "state.json")
    print(f"run_id: {state['job_id']}")
    print(f"status: {state['status']}")
    print(f"goal: {state['goal']}")
    print(f"iteration: {state['iteration']}/{state['max_iterations']}")
    print(f"worker_thread_id: {state.get('worker_thread_id') or 'none'}")
    if state["iterations"]:
        latest = state["iterations"][-1]
        print(f"last_decision: {latest['decision']} ({latest['decision_reason']})")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.job_id)
    state = load_json(run_dir / "state.json")
    iterations = state["iterations"]
    if args.last:
        iterations = iterations[-args.last :]
    for iteration in iterations:
        print(f"=== iteration {iteration['iteration']} ===")
        print(f"decision: {iteration['decision']} ({iteration['decision_reason']})")
        if iteration["worker_text"]:
            print("worker:")
            print(iteration["worker_text"])
        print("verification:")
        print(build_verification_summary(iteration["verification_results"], iteration["success_file_results"]))
        print("")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal orchestrator loop for local Codex worker sessions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new orchestration run")
    run_parser.add_argument("--goal")
    run_parser.add_argument("--job-file")
    run_parser.add_argument("--context-file")
    run_parser.add_argument("--verify", action="append")
    run_parser.add_argument("--success-file-pattern", action="append")
    run_parser.add_argument("--max-iterations", type=int)
    run_parser.add_argument("--max-consecutive-identical-failures", type=int)
    run_parser.add_argument("--verify-timeout-seconds", type=int)
    run_parser.add_argument("--worker-model")
    run_parser.set_defaults(func=cmd_run)

    resume_parser = subparsers.add_parser("resume", help="Resume an orchestration run")
    resume_parser.add_argument("job_id", nargs="?")
    resume_parser.add_argument("--force", action="store_true")
    resume_parser.set_defaults(func=cmd_resume)

    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("job_id", nargs="?")
    status_parser.set_defaults(func=cmd_status)

    logs_parser = subparsers.add_parser("logs", help="Print run logs")
    logs_parser.add_argument("job_id", nargs="?")
    logs_parser.add_argument("--last", type=int)
    logs_parser.set_defaults(func=cmd_logs)

    return parser


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
