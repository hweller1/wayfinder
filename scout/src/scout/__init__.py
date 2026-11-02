"""scout: indexed semantic code exploration. `Scout` (the SDK) is exported lazily so importing the package
stays cheap and pulls in no adapters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scout.surfaces.sdk import Scout

__all__ = ["Scout"]


def __getattr__(name: str) -> Any:
    if name == "Scout":
        from scout.surfaces.sdk import Scout

        return Scout
    raise AttributeError(f"module 'scout' has no attribute {name!r}")
