"""Group overlap between training and validation, and output anonymization."""

from evaluation.common import anonymize
from evaluation.data_splitting.group_overlap import group_overlap


def test_overlap_counts_validation_samples_whose_group_is_in_training():
    r = group_overlap(['a', 'a', 'b'], ['a', 'c', 'c', 'b'])
    assert r['val_samples'] == 4 and r['val_groups'] == 3
    assert r['val_samples_seen_in_train'] == 2 and r['val_groups_seen_in_train'] == 2
    assert r['share_val_samples_seen'] == 0.5
    assert r['share_val_groups_seen'] == round(2 / 3, 4)


def test_a_group_split_has_no_overlap():
    r = group_overlap(['a', 'b', 'b'], ['c', 'd'])
    assert r['val_samples_seen_in_train'] == 0 and r['share_val_samples_seen'] == 0.0


def test_empty_validation_has_no_share():
    assert group_overlap(['a'], [])['share_val_samples_seen'] is None


def test_anonymize_replaces_ids_in_keys_and_values():
    out = anonymize({'per_video': {'real1': {'fold': 0}}, 'videos': ['real2', 'other'], 'video': 'real1', 'n': 3}, {'real1': 'V01', 'real2': 'V02'})
    assert out == {'per_video': {'V01': {'fold': 0}}, 'videos': ['V02', 'other'], 'video': 'V01', 'n': 3}
