"""Display assets that the terminal and the browser front ends share.

``sprites`` holds the PNG codec and cell renderer, ``scene_art`` prepares the
character fixtures, and ``rules_text`` holds the player-facing rules. None of
them reads or changes game state, and none imports a front end.
"""
