from summarize_libero import summarize


def test_summarize_preserves_every_failed_episode():
    result = summarize(
        {
            "overall": {"pc_success": 50.0},
            "per_task": [
                {
                    "task_group": "libero_spatial",
                    "task_id": 3,
                    "metrics": {"successes": [True, False, False, True]},
                }
            ],
        }
    )
    assert result["tasks"][0]["success_rate"] == 0.5
    assert [item["episode_index"] for item in result["unresolved_failures"]] == [1, 2]

