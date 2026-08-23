from run_task_skill_validation import _select_states


def test_unseen_state_selection_is_deterministic_and_excludes_development():
    first = _select_states(
        total=20, count=3, excluded={7, 11}, program_sha256="a" * 64,
    )
    second = _select_states(
        total=20, count=3, excluded={7, 11}, program_sha256="a" * 64,
    )
    assert first == second
    assert len(first) == len(set(first)) == 3
    assert not {7, 11}.intersection(first)
