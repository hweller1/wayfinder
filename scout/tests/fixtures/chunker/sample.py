"""Fixture: top-level declarations of a Python module."""

import os


def load_settings(path, defaults=None):
    """Read key=value lines from path and merge them over the defaults mapping."""
    settings = dict(defaults or {})
    with open(os.fspath(path), encoding="utf-8") as handle:
        for raw in handle:
            key, _, value = raw.partition("=")
            settings[key.strip()] = value.strip()
    return settings


class Registry:
    """Keeps named handlers and calls them in registration order."""

    def __init__(self):
        self._handlers = []

    def register(self, name, handler):
        self._handlers.append((name, handler))
        return handler

    def dispatch(self, event):
        return [handler(event) for _, handler in self._handlers]


@staticmethod
def describe(registry):
    """Return a short human readable summary of every registered handler name."""
    names = [name for name, _ in registry._handlers]
    return ", ".join(names) if names else "no handlers registered"
