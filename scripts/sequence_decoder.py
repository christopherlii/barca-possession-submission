"""Frozen submission inference: sequence decoder."""
import xml.etree.ElementTree as ET


HOME, AWAY, DEAD, UNKNOWN = "H", "A", "D", "X"


def team_of_each_track(meta_path):
    """List over track slots: H or A. First team in the ATD metadata is home, as in the baseline."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return [HOME if team_id == team_ids[0] else AWAY for team_id in slots]

