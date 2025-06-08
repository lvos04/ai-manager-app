import re


def is_valid_license_plate(plate: str) -> bool:
    """Check if a string is a valid license plate number.

    A valid license plate has the format ``ABC1234`` where ``ABC`` are uppercase
    letters and ``1234`` are digits.

    Parameters
    ----------
    plate : str
        License plate string to validate.

    Returns
    -------
    bool
        ``True`` if ``plate`` matches the required format, otherwise ``False``.
    """
    return bool(re.fullmatch(r"[A-Z]{3}\d{4}", plate))

