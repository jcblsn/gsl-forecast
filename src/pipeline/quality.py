"""Normalize USGS approval status and qualifier codes for daily values.

Historical rows use both single-letter and full-word approval labels. These helpers map
both forms to one representation before the pipeline counts provisional or qualified days.
"""

APPROVED = "approved"
PROVISIONAL = "provisional"
UNKNOWN = "unknown"

# Qualifier codes retained as data-quality context beside the approval status.
QUALIFIER_CODES = ("estimated", "ice", "equip", "revised", "forceinterpolation")

_APPROVAL_WORDS = {
    "a": APPROVED,
    "approved": APPROVED,
    "p": PROVISIONAL,
    "provisional": PROVISIONAL,
}


def normalize_flags(approval: str | None, qualifiers: list[str] | str | None) -> str:
    """Return a lowercase approval label followed by any qualifier codes."""
    codes = qualifiers if isinstance(qualifiers, list) else [qualifiers] if qualifiers else []
    parts = [_APPROVAL_WORDS.get(str(approval or "").strip().lower(), UNKNOWN)]
    parts += [str(c).strip().lower() for c in codes if str(c).strip()]
    return ",".join(parts)


def approval_sql(column: str) -> str:
    """SQL that reads the approval status out of a flag string, in either vocabulary."""
    low = f"LOWER(COALESCE({column}, ''))"
    return f"""CASE
        WHEN {low} LIKE '%provisional%' OR {low} = 'p' OR {low} LIKE 'p,%' THEN '{PROVISIONAL}'
        WHEN {low} LIKE '%approved%' OR {low} = 'a' OR {low} LIKE 'a,%' THEN '{APPROVED}'
        ELSE '{UNKNOWN}'
    END"""


def is_provisional_sql(column: str) -> str:
    return f"{approval_sql(column)} = '{PROVISIONAL}'"


def has_qualifier_sql(column: str, code: str) -> str:
    """SQL that is true when one qualifier code is present in a flag string."""
    if code not in QUALIFIER_CODES:
        raise ValueError(f"Unknown qualifier code {code}; expected one of {QUALIFIER_CODES}")
    return f"LOWER(COALESCE({column}, '')) LIKE '%{code}%'"
