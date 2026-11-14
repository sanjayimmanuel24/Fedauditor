"""Create a known demo login for demonstrations and marking sessions.

The application ships no built-in account: rows in ``users`` only appear after
using the signup form, so a deleted ``users.db`` (or a fresh checkout) leaves
you with no way in. This script creates one account idempotently.

    python seed_demo.py
    python seed_demo.py --email demo@fedhealth.local --password demo12345
    python seed_demo.py --reset          # reset the password if it is forgotten

Exit codes: 0 on success (created, already present, or updated), 1 on error.
"""

from __future__ import annotations

import argparse
import sys

from database import create_user, init_db, set_password, verify_user

DEFAULT_NAME = "Demo Clinician"
DEFAULT_EMAIL = "demo@fedhealth.local"
DEFAULT_PASSWORD = "demo12345"
MIN_PASSWORD_LENGTH = 8  # mirrors the signup route


def seed(name: str, email: str, password: str, reset: bool = False) -> str:
    """Create the account, or report that it already exists.

    Returns one of ``created``, ``exists``, ``updated``, ``missing``.
    """
    init_db()
    if create_user(name, email, password):
        return "created"
    if reset:
        return "updated" if set_password(email, password) else "missing"
    return "exists"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a known demo login.")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="if the account already exists, overwrite its password",
    )
    args = parser.parse_args()

    # The login route lowercases emails, so normalise here or the account
    # would exist but never match.
    name = args.name.strip()
    email = args.email.strip().lower()
    password = args.password

    if not name:
        print("error: name must not be empty", file=sys.stderr)
        return 1
    if not email or "@" not in email:
        print(f"error: invalid email {email!r}", file=sys.stderr)
        return 1
    if len(password) < MIN_PASSWORD_LENGTH:
        print(
            f"error: password must be at least {MIN_PASSWORD_LENGTH} characters "
            "(same rule as the signup form)",
            file=sys.stderr,
        )
        return 1

    try:
        status = seed(name, email, password, reset=args.reset)
    except Exception as exc:  # pragma: no cover - surfaced to the operator
        print(f"error: could not seed account: {exc}", file=sys.stderr)
        return 1

    if status == "missing":
        print(f"error: {email} does not exist, nothing to reset", file=sys.stderr)
        return 1

    # Confirm the credentials actually authenticate before claiming success.
    if verify_user(email, password) is None:
        print(
            "error: account exists but the password does not verify. "
            "Re-run with --reset.",
            file=sys.stderr,
        )
        return 1

    labels = {
        "created": "created",
        "exists": "already existed (password unchanged)",
        "updated": "password reset",
    }
    print("Demo account ready.")
    print(f"  email:    {email}")
    print(f"  password: {password}")
    print(f"  status:   {labels[status]}")
    print("\nOpen http://127.0.0.1:5000/login and sign in.")
    if status == "exists":
        print("(Use --reset if this password is not the one on record.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
