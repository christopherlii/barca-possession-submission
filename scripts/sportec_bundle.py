"""Convert permitted public Sportec other-match tracking into separate inputs/targets.

Native provider labels are an external benchmark, not the challenge answer key.
BallStatus/BallPossession are target-only and never written to model inputs.
Coordinates use Sportec's centre-origin metric pitch. No upstream code copied.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from path_guard import check_paths

PERIODS = {'firstHalf': 1, 'secondHalf': 2}


def open_xml(path):
    return gzip.open(path, 'rb') if str(path).endswith('.gz') else open(path, 'rb')


def frames(path, expected_match=None):
    """Yield (period, team, person, attributes); streaming memory bounded by frameset."""
    with open_xml(path) as handle:
        section, team, person = None, None, None
        frameset = None
        for event, element in ET.iterparse(handle, events=('start', 'end')):
            if event == 'start' and element.tag == 'FrameSet':
                if expected_match is not None and not str(element.get('MatchId','')).endswith(expected_match):
                    raise ValueError('tracking match differs from declared partition/metadata')
                section = PERIODS.get(element.get('GameSection'))
                team, person = element.get('TeamId'), element.get('PersonId')
                frameset = element
            elif event == 'end' and element.tag == 'Frame':
                if section is not None:
                    yield section, team, person, dict(element.attrib)
                element.clear()
                if frameset is not None:
                    frameset.clear()
            elif event == 'end' and element.tag == 'FrameSet':
                element.clear()
                frameset = None


def target(status, possession):
    if status == '0':
        return 'D'
    if status == '1' and possession in ('1', '2'):
        return 'H' if possession == '1' else 'A'
    return 'X'


def convert(tracking, metadata, out):
    check_paths([tracking, metadata, out])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = ET.parse(metadata).getroot()
    general = root.find('.//General')
    environment = root.find('.//Environment')
    home, away = general.get('HomeTeamId'), general.get('GuestTeamId')
    if not home or not away or home == away:
        raise ValueError('metadata needs distinct home/away teams')
    pitch = (float(environment.get('PitchX')), float(environment.get('PitchY')))
    if not all(50 < value < 150 for value in pitch):
        raise ValueError('invalid metric pitch')
    roster = []
    for team in root.iter('Team'):
        if team.get('TeamId') not in (home, away):
            continue
        label = 'H' if team.get('TeamId') == home else 'A'
        roster.extend((p.get('PersonId'), label) for p in team.iter('Player'))
    if len({p for p, _ in roster}) != len(roster) or not roster:
        raise ValueError('empty or repeated player roster')
    slots = {p: i for i, (p, _) in enumerate(roster)}
    ball = {}
    for period, team, _, row in frames(tracking, general.get('MatchId')):
        if team == 'BALL':
            key = (period, int(row['N']))
            if key in ball:
                raise ValueError('repeated ball frame')
            ball[key] = row
    blocks = []
    keys = []
    for period in (1, 2):
        ids = sorted(n for h, n in ball if h == period)
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError('require complete native ball timeline in each half')
        # The official scorer discovers graded runs from target X separators.
        # Keep a real ungraded gap so same-team spells cannot merge across halves.
        if keys:
            keys.append(None)
        first = len(keys)
        keys.extend((period, n) for n in ids)
        blocks.append((period, first, len(keys) - 1))
    index = {key: i for i, key in enumerate(keys) if key is not None}
    players = np.full((len(keys), len(roster), 2), np.nan, dtype=np.float32)
    for period, team, person, row in frames(tracking, general.get('MatchId')):
        if team == 'BALL' or person not in slots:
            continue
        frame = index.get((period, int(row['N'])))
        if frame is not None:
            players[frame, slots[person]] = float(row['X']), float(row['Y'])
    # Canonical orientation must agree with the synthetic/ATD feature convention.
    # Use declared goalkeeper identity and observed positions, never possession.
    keepers=[p.get('PersonId') for t in root.iter('Team') if t.get('TeamId')==home
             for p in t.iter('Player') if p.get('PlayingPosition')=='TW']
    keeper_slots=[slots[p] for p in keepers if p in slots]
    first,last=blocks[0][1:]
    keeper_x=players[first:last+1,keeper_slots,0] if keeper_slots else np.array([])
    keeper_x=keeper_x[np.isfinite(keeper_x)]
    if not len(keeper_x):
        raise ValueError('cannot orient match without declared home goalkeeper observations')
    median_x=float(np.median(keeper_x))
    if abs(median_x)<pitch[0]*.1:
        raise ValueError('ambiguous goalkeeper orientation')
    sign=-1 if median_x>0 else 1
    players*=sign
    paths = {k: str((out / name).resolve()) for k, name in (
        ('atd_raw', 'tracking.txt'), ('atd_meta', 'metadata.xml'), ('gt', 'targets.csv'),
        ('latent', 'diagnostics.csv'), ('tagging', 'tagging.json'))}
    counts = Counter()
    with open(paths['atd_raw'], 'w') as observed, open(paths['gt'], 'w') as truth:
        truth.write('frame,label\n')
        for t, key in enumerate(keys):
            if key is None:
                observed.write(f'{t+1}:'+';'.join(['NaN,NaN,NaN']*len(roster))+':NaN,NaN,NaN\n')
                truth.write(f'{t},X\n')
                counts['X'] += 1
                continue
            row = ball[key]
            points = (players[t] + np.array(pitch) / 2.) / np.array(pitch)
            groups = ['NaN,NaN,NaN' if not np.isfinite(point).all() else
                      f'{point[0]:.8f},{point[1]:.8f},0' for point in points]
            point = (sign*np.array([float(row['X']), float(row['Y'])]) + np.array(pitch)/2.) / np.array(pitch)
            ball_text = 'NaN,NaN,NaN' if not np.isfinite(point).all() else f'{point[0]:.8f},{point[1]:.8f},0'
            observed.write(f'{t+1}:'+ ';'.join(groups) + ':' + ball_text + '\n')
            label = target(row.get('BallStatus'), row.get('BallPossession'))
            counts[label] += 1
            truth.write(f'{t},{label}\n')
    meta = ET.Element('main')
    md = ET.SubElement(meta, 'Metadata')
    teams = ET.SubElement(md, 'Teams')
    for label in ('H', 'A'):
        ET.SubElement(ET.SubElement(teams, 'Team', id=label), 'Name').text = label
    people = ET.SubElement(md, 'Players')
    for i, (_, label) in enumerate(roster):
        player = ET.SubElement(people, 'Player', id=f'P{i+1}', teamId=label)
        ET.SubElement(player, 'Name').text = f'Track {i+1}'
        ET.SubElement(player, 'ShirtNumber').text = str(i+1)
    ET.ElementTree(meta).write(paths['atd_meta'], encoding='utf-8', xml_declaration=True)
    # Bundle diagnostics schema only; neither field is supplied to the predictor.
    with open(paths['latent'], 'w') as handle:
        handle.write('frame,cause,ball_missing\n')
        for t, key in enumerate(keys):
            if key is None:
                handle.write(f'{t},ungraded,1\n')
                continue
            row = ball[key]
            missing = not all(np.isfinite(float(row[c])) for c in ('X','Y'))
            handle.write(f'{t},other_match,{int(missing)}\n')
    Path(paths['tagging']).write_text('[]\n')
    manifest = dict(format='sportec-native-targets-v3', match_id=general.get('MatchId'),
        seed=general.get('MatchId'),
        n_frames=len(keys), fps=25, pitch=pitch, blocks=blocks, paths=paths,
        label_counts=dict(counts), orientation='home_first_half_left',
        rotation_degrees=180 if sign<0 else 0,
        orientation_basis='declared home goalkeeper median first-half x',
        limitation='Provider native possession/status, pending independent boundary audit. '
                   'Do not claim challenge or human-audited possession accuracy.',
        source_sha256={str(Path(p).name):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (tracking, metadata)})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = convert(args.tracking, args.metadata, args.out)
    print(json.dumps({k:result[k] for k in ('n_frames','blocks','label_counts','limitation')}))
