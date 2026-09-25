"""Certified policy controllers layered over exact DTH stage games.

This package imports nothing. Import each provider from its own module, such
as ``arena.policies.perfect_hal``, so a runtime that serves one provider never
loads torch, gymnasium, or stable-baselines3 through a sibling module.
"""
