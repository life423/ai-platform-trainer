"""Canonical, headless Pixel Pursuit arena.

The single source of truth for the game rules that learned agents are
trained, evaluated and deployed against:

- config: every speed, size and timer, mirroring Play mode
- sim / episode: a pygame-free, frame-accurate simulation of Play mode
- observations: the shared observation and action encoders that training
  and in-game inference both call
- player_bots / enemy_policies: scripted opponents and baseline enemies
- game_bridge: snapshots of the live pygame entities for in-game inference

Nothing here imports pygame at module level, so it runs headless and fast.
"""
