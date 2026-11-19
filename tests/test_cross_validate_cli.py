import pytest

from scripts import cross_validate as cv


@pytest.mark.parametrize('command', ['fold', 'final'])
def test_cli_accepts_explicit_training_seed_and_device(monkeypatch, command):
    captured = []
    args = ['cross_validate.py', command]
    args += ['0', '--root', 'runs/test'] if command == 'fold' else ['--out', 'runs/test']
    args += ['--seed', '2', '--device', 'cpu']
    monkeypatch.setattr('sys.argv', args)
    monkeypatch.setattr(cv, 'run_fold', captured.append)
    monkeypatch.setattr(cv, 'run_final', captured.append)
    cv.main()
    assert captured[0].seed == 2
    assert captured[0].device == 'cpu'


def test_training_forwards_seed_device_and_exclusions(monkeypatch):
    captured = {}

    def fake_train(features, out, exclude, val, **kwargs):
        captured.update(features=features, out=out, exclude=exclude, val=val, **kwargs)

    monkeypatch.setattr(cv.tp, 'train', fake_train)
    cv.train('features', 'output', ['held-video'], 12, seed=2, device='cpu')
    assert captured['seed'] == 2
    assert captured['device'] == 'cpu'
    assert captured['exclude'] == ['held-video']
    assert captured['steps'] == 12
