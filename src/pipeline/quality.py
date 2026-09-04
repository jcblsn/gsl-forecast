"""Approval status and qualifier codes for USGS daily values.

USGS has used two vocabularies in the record this project holds. Older rows carry the single
letters `A` and `P`. Newer rows carry the words `Approved` and `Provisional`. A pattern match
on one word therefore misses the other, and the count of provisional days becomes wrong
without an error. These helpers give one definition of each flag, and every reader uses it.
"""

APPROVED = "approved"
PROVISIONAL = "provisional"
UNKNOWN = "unknown"

# Qualifier codes USGS attaches beside the approval status. Each one says the value is less
# trustworthy than a plain measurement, so each one is counted.
QUALIFIER_CODES = ("estimated", "ice", "equip", "revised", "forceinterpolation")

_APPROVAL_WORDS = {
    "a": APPROVED,
    "approved": APPROVED,
    "p": PROVISIONAL,
    "provisional": PROVISIONAL,
}


def normalize_flags(approval: str | None, qualifiers: list[str] | str | None) -> str:
    """The canonical flag string for one daily value.

    The approval status comes first as a full word, then the qualifier codes in lower case.
    Both vocabularies map to the same output, so a row ingested today and the same row
    ingested in 2019 compare equal.
    """
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
