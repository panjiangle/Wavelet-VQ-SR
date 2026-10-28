"""Lean package initializer for pinned FeMaSR architectures.

The clean checkout's initializer eagerly imports the entire training stack,
including optional pyIQA. Our common research harness directly imports the
architecture and owns its training/evaluation loop. This initializer changes
import behavior only; no FeMaSR architecture file is modified.
"""
