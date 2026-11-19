"""Group-aware validation splitting.

The bug these guard against is silent and expensive: with a random split over
sliding windows, nearly every validation window shares frames with a training
window, so the reported accuracy measures memorization. These lock in that (1) no
group ever lands on both sides, (2) the split is deterministic for a seed, and (3)
adding data to one group does not reshuffle the others.
"""

from breakpoint_vision.data.split import group_sizes, group_split, split_groups


def test_no_group_appears_on_both_sides():
    groups = [f"clip{i // 20}" for i in range(200)]
    train_idx, val_idx = group_split(groups, 0.2, seed=7)

    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    assert train_groups & val_groups == set()
    assert train_groups | val_groups == set(groups)
    assert len(train_idx) + len(val_idx) == len(groups)


def test_split_is_deterministic_for_a_seed():
    groups = [f"clip{i}" for i in range(30)]
    a = split_groups(groups, 0.3, seed=99)
    b = split_groups(groups, 0.3, seed=99)
    assert a == b
    assert split_groups(groups, 0.3, seed=100) != a


def test_adding_samples_to_one_group_does_not_move_other_groups():
    """Index shuffling would reshuffle everything; hashed group ranking does not."""
    before = split_groups([f"clip{i}" for i in range(20)], 0.2, seed=3)
    after = split_groups([f"clip{i}" for i in range(20)] + ["clip0"] * 50, 0.2, seed=3)
    assert before == after


def test_both_sides_are_non_empty_even_at_extreme_fractions():
    groups = ["a", "b", "c", "d"]
    for fraction in (0.0, 0.01, 0.99, 1.0):
        train, val = split_groups(groups, fraction, seed=1)
        assert train and val, fraction
        assert set(train) | set(val) == set(groups)


def test_a_single_group_cannot_be_split():
    """One clip means there is nothing to hold out; callers must not get a leak."""
    train, val = split_groups(["only"] * 10, 0.2, seed=1)
    assert train == ["only"]
    assert val == []


def test_group_sizes_counts_samples_per_group():
    assert group_sizes(["a", "b", "a", "a"]) == {"a": 3, "b": 1}
