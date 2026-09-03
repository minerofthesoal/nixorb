"""The version in the source has to be the version being released.

`release.yml` rewrites `nixorb/__init__.py` from the git tag, so a tag put
on the wrong commit does not fail the build — it publishes older code
wearing a newer number. That is not hypothetical: v2.0.10 and v2.0.11 were
placed on the same commit, PyPI's 2.0.11 shipped 2.0.10's code, and a bug
fixed in 2.0.11 survived what looked like an upgrade.

The release workflow now refuses a tag that disagrees with the source.
These tests keep the source itself self-consistent, so the disagreement
cannot be introduced in the first place.
"""
from __future__ import annotations

import re
from pathlib import Path

import nixorb

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
HEADING = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)


def changelog_versions() -> list[str]:
    return HEADING.findall(CHANGELOG.read_text())


def test_the_changelog_documents_this_version():
    # The release workflow refuses to publish a version with no notes.
    assert nixorb.__version__ in changelog_versions(), (
        f"CHANGELOG.md has no '## [{nixorb.__version__}]' section"
    )


def test_this_version_is_the_newest_entry():
    # A bump without a matching entry at the top means either the notes or
    # the bump was forgotten — and a tag on that commit ships a mismatch.
    versions = changelog_versions()
    assert versions, "CHANGELOG.md has no version headings at all"
    assert versions[0] == nixorb.__version__, (
        f"__version__ is {nixorb.__version__} but the newest CHANGELOG "
        f"entry is {versions[0]}"
    )


def test_no_version_is_documented_twice():
    versions = changelog_versions()
    duplicates = {v for v in versions if versions.count(v) > 1}
    assert not duplicates, f"CHANGELOG.md documents these twice: {duplicates}"


def test_the_version_is_a_plain_release_number():
    # hatchling reads this string; anything it cannot parse becomes a
    # confusing failure at build time rather than here.
    assert re.fullmatch(r"\d+\.\d+\.\d+", nixorb.__version__), nixorb.__version__


def test_pyproject_takes_its_version_from_the_source():
    # If this stops being true, the workflow's guard is checking a file
    # that no longer decides the published version.
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "nixorb/__init__.py"' in pyproject
