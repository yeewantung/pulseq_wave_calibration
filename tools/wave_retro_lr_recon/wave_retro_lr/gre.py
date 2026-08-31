"""Reserved GRE acquisition adapter boundary."""

from __future__ import annotations

from typing import NoReturn


def prepare_gre(*args: object, **kwargs: object) -> NoReturn:
    """Reject GRE until the measured MPRAGE workflow is user-validated.

    Args:
        *args: Reserved future GRE positional inputs; currently rejected.
        **kwargs: Reserved future GRE keyword inputs; currently rejected.

    Returns:
        This function never returns and always raises ``NotImplementedError``.
    """

    raise NotImplementedError(
        "GRE support is deferred until the MPRAGE cleanup passes real-data validation."
    )
