"""Check published aggregates and local document links without private data."""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / 'results' / 'tables' / name).open(newline='') as source:
        return list(csv.DictReader(source))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_tables():
    ball = rows('ball_per_video.csv')
    totals = Counter()
    require(len(ball) == 7 and len({r['video'] for r in ball}) == 7, 'Ball video IDs')
    for row in ball:
        counts = {key: int(value) for key, value in row.items() if key != 'video'}
        require(all(value >= 0 for value in counts.values()), 'Negative ball count')
        require(counts['within_3px'] <= counts['within_8px'] <= counts['visible'], 'Ball hit ordering')
        require(counts['within_8px'] + counts['no_output'] + counts['wrong_over_8px'] == counts['visible'], 'Ball visible partition')
        require(counts['false_alarms'] <= counts['no_ball_frames'], 'Ball false-alarm denominator')
        require(counts['visible'] + counts['no_ball_frames'] + counts['uncertain'] == 50, 'Ball frame sample')
        totals.update(counts)
    for key, expected in dict(visible=248, within_8px=218, within_3px=204, no_output=23, wrong_over_8px=7, false_alarms=20, no_ball_frames=88, uncertain=14).items():
        require(totals[key] == expected, f'Ball headline: {key}')

    points = rows('points_per_video.csv')
    folds = rows('point_folds.csv')
    require(len(points) == len(folds) == 42, 'Point video count')
    require(len({r['video'] for r in points}) == 42, 'Duplicate point video')
    require({(r['video'], r['fold'], r['lighting']) for r in points} == {(r['video'], r['fold'], r['lighting']) for r in folds}, 'Fold mapping drift')
    require(Counter(r['fold'] for r in points) == {'0': 11, '1': 11, '2': 10, '3': 10}, 'Fold sizes')
    require(sum(int(r['labeled_segments']) for r in points) == 1351, 'Point denominator')
    kept = 0
    for row in points:
        n = int(row['labeled_segments'])
        require(bool(row['kept_whole']) == bool(n), 'Missing point retention')
        require(0 <= float(row['dead_time_removed']) <= 1, 'Dead-time fraction')
        if n:
            value = float(row['kept_whole'])
            require(0 <= value <= 1, 'Retention fraction')
            count = round(n * value)
            require(abs(count / n - value) <= 0.000051, 'Rounded point fraction inconsistent with an integer count')
            kept += count
    require(kept == 1272, 'Original-selection retained count')
    require(round(sum(float(r['dead_time_removed']) for r in points) / 42, 3) == 0.504, 'Mean dead-time headline')
    seeds = rows('points_seeds.csv')
    require([int(r['seed']) for r in seeds] == [0, 1, 2], 'Seed identities')
    require([float(r['kept_whole_nested']) for r in seeds] == [0.919, 0.924, 0.920], 'Held-fold retention')
    require([float(r['dead_removed_mean_nested']) for r in seeds] == [0.524, 0.505, 0.512], 'Held-fold dead time')

    formats = rows('ball_formats.csv')
    variants = defaultdict(set)
    for row in formats:
        variants[row['video']].add(row['variant'])
    complete = {video for video, kinds in variants.items() if len(kinds) == 6}
    require(len(complete) == 6, 'Complete format-study videos')
    pooled = defaultdict(Counter)
    for row in formats:
        if row['video'] in complete:
            pooled[row['variant'], row['resize']].update({k: int(row[k]) for k in ('visible', 'within_8px', 'within_3px')})
    published = rows('ball_formats_pooled.csv')
    require(len(published) == len(pooled) == 12, 'Pooled format rows')
    for row in published:
        require(dict(pooled[row['variant'], row['resize']]) == {k: int(row[k]) for k in ('visible', 'within_8px', 'within_3px')}, 'Pooled format mismatch')
    court = rows('court.csv')
    require([int(r['frames']) for r in court] == [200, 359, 359, 359], 'Court denominators')
    require([float(r['within_15cm']) for r in court] == [0.975, 0.251, 0.535, 0.619], 'Court headline values')
    demo = {r['metric']: float(r['value']) for r in rows('match_demo.csv')}
    require(demo['labeled_serve_segments'] == 42 and demo['kept_whole'] == 1, 'Demonstration retention')
    require(demo['dead_time_removed_time_weighted'] == 0.546, 'Demonstration dead time')
    return {'ball_visible_hits': totals['within_8px'], 'ball_visible_frames': totals['visible'], 'point_segments': 1351, 'original_selection_kept': kept, 'point_videos': len(points), 'seeds': len(seeds)}


def check_documents():
    for path in [ROOT / 'README.md', *ROOT.glob('docs/*.md'), *ROOT.glob('results/*.md'), ROOT / 'evaluation' / 'README.md']:
        text = path.read_text()
        require('\u2014' not in text and '\u2013' not in text, f'Prohibited dash: {path.name}')
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', text):
            if '://' in target or target.startswith(('mailto:', '#')):
                continue
            local = unquote(target.split('#')[0])
            require((path.parent / local).exists(), f'Broken link in {path.name}: {target}')
    source = (ROOT / 'report' / 'report.typ').read_text()
    bibliography = (ROOT / 'report' / 'references.bib').read_text()
    known = set(re.findall(r'<([\w-]+)>', source)) | set(re.findall(r'@\w+\{([^,]+),', bibliography))
    require(set(re.findall(r'@([\w-]+)', source)) <= known, 'Unresolved report reference')
    for name in ('report', 'summary'):
        require((ROOT / 'report' / f'{name}.pdf').read_bytes().startswith(b'%PDF-'), f'Missing PDF: {name}')


def main():
    result = check_tables()
    check_documents()
    print(json.dumps({'status': 'pass', 'checks': 'published table consistency, local links, report references, PDF presence', **result}, indent=2))


if __name__ == '__main__':
    main()
