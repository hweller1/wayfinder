"""Shared live-test types (task 7.10)."""

from dataclasses import dataclass

from scout.adapters.turbopuffer import TurbopufferStore


@dataclass(frozen=True)
class LiveStore:
    store: TurbopufferStore
    prefix: str  # `scout_dev_suite_<8hex>_`: every namespace of a test (and `__tree` siblings) starts here
