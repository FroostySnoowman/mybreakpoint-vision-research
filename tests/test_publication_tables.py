import shutil

import pytest

from scripts import check_publication as publication


def test_published_tables_reconcile():
    result = publication.check_tables()
    assert result['point_segments'] == 1351
    assert result['ball_visible_frames'] == 248


def test_checker_rejects_changed_localization_denominator(tmp_path, monkeypatch):
    shutil.copytree(publication.ROOT / 'results' / 'tables', tmp_path / 'results' / 'tables')
    path = tmp_path / 'results' / 'tables' / 'ball_per_video.csv'
    path.write_text(path.read_text().replace('V02,41,', 'V02,42,'))
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Ball visible partition'):
        publication.check_tables()


def test_checker_rejects_fold_assignment_drift(tmp_path, monkeypatch):
    shutil.copytree(publication.ROOT / 'results' / 'tables', tmp_path / 'results' / 'tables')
    path = tmp_path / 'results' / 'tables' / 'point_folds.csv'
    path.write_text(path.read_text().replace('V01,day,0', 'V01,day,1'))
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Fold mapping drift'):
        publication.check_tables()
