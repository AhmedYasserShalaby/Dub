import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrate import (
    build_failure_signature,
    build_initial_prompt,
    build_retry_prompt,
    build_success_file_results,
    build_verification_summary,
    count_consecutive_identical_failures,
    parse_codex_output,
    run_shell_command,
    verification_failed,
)


class OrchestrateTests(unittest.TestCase):
    def test_parse_codex_output_uses_last_message_file(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="orchestrate-test-"))
        jsonl_path = temp_dir / "worker.jsonl"
        last_message_path = temp_dir / "last_message.txt"
        jsonl_path.write_text(
            '\n'.join(
                [
                    '{"type":"thread.started","thread_id":"abc-123"}',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"fallback"}}',
                ]
            ),
            encoding="utf-8",
        )
        last_message_path.write_text("final answer", encoding="utf-8")

        thread_id, message = parse_codex_output(jsonl_path, last_message_path)

        self.assertEqual(thread_id, "abc-123")
        self.assertEqual(message, "final answer")

    def test_failure_signature_is_stable_for_same_failure(self):
        verification = [
            {
                "command": "pytest -q",
                "exit_code": 1,
                "stdout": "FAILED tests/test_x.py::test_y - AssertionError: no\n",
                "stderr": "",
            }
        ]
        sig1 = build_failure_signature(verification, " file.py | 2 +-", "I changed file.py")
        sig2 = build_failure_signature(verification, " file.py | 2 +-", "I changed file.py")
        self.assertEqual(sig1, sig2)

    def test_count_consecutive_identical_failures(self):
        state = {
            "iterations": [
                {"failure_signature": "aaa", "decision": "retry"},
                {"failure_signature": "aaa", "decision": "retry"},
                {"failure_signature": "bbb", "decision": "retry"},
            ]
        }
        self.assertEqual(count_consecutive_identical_failures(state, "bbb"), 1)
        self.assertEqual(count_consecutive_identical_failures(state, "aaa"), 0)

    def test_verification_summary_includes_commands_and_files(self):
        summary = build_verification_summary(
            [
                {
                    "command": "pytest -q",
                    "exit_code": 1,
                    "stdout": "one\n",
                    "stderr": "two\n",
                }
            ],
            [{"pattern": "dist/*.mp4", "ok": False, "matches": []}],
        )
        self.assertIn("$ pytest -q", summary)
        self.assertIn("dist/*.mp4: missing", summary)

    def test_prompt_builders_include_goal_and_feedback(self):
        initial = build_initial_prompt(
            goal="Ship feature X",
            context_text="Use the repo style",
            verify_commands=["pytest -q"],
            success_patterns=["dist/*.mp4"],
        )
        retry = build_retry_prompt(
            goal="Ship feature X",
            prior_worker_text="Changed foo.py",
            verification_summary="pytest failed",
            git_summary="git diff --stat:\nfoo.py | 2 +-",
            consecutive_failures=1,
        )
        self.assertIn("Ship feature X", initial)
        self.assertIn("pytest -q", initial)
        self.assertIn("Changed foo.py", retry)
        self.assertIn("pytest failed", retry)

    def test_success_file_results_and_verification_failed(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="orchestrate-test-"))
        (temp_dir / "artifact.txt").write_text("ok", encoding="utf-8")
        success = build_success_file_results(["*.txt", "*.mp4"], temp_dir)
        self.assertTrue(success[0]["ok"])
        self.assertFalse(success[1]["ok"])
        self.assertTrue(verification_failed([], success))
        self.assertFalse(verification_failed([], [success[0]]))

    def test_run_shell_command_marks_timeout(self):
        with patch("orchestrate.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=3)):
            result = run_shell_command("sleep 10", cwd=Path("."), timeout_seconds=3)
        self.assertEqual(result["exit_code"], 124)
        self.assertTrue(result["timed_out"])


if __name__ == "__main__":
    unittest.main()
