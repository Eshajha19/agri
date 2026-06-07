import pytest

from backend.routers import knowledge


@pytest.mark.parametrize(
    ("raw_region", "expected"),
    [
        ("northwest", "northwest"),
        ("  NORTHWEST  ", "northwest"),
        ("Punjab", "northwest"),
        (" uttar pradesh ", "northeast"),
        ("   maharashtra   ", "central"),
        ("KERALA", "southwest"),
        ("unknown-region", "central"),
        ("", "central"),
        ("   ", "central"),
    ],
)
def test_resolve_region_canonicalizes_aliases_and_defaults(raw_region, expected):
    assert knowledge._resolve_region(raw_region) == expected


@pytest.mark.parametrize(
    ("raw_season", "expected"),
    [
        ("kharif", "kharif"),
        ("  KHARIF  ", "kharif"),
        ("winter", "rabi"),
        (" post-monsoon ", "rabi"),
        ("PRE-KHARIF", "zaid"),
        ("spring", "zaid"),
        ("unknown-season", "kharif"),
        ("", "kharif"),
        ("   ", "kharif"),
    ],
)
def test_resolve_season_canonicalizes_synonyms_and_defaults(raw_season, expected):
    assert knowledge._resolve_season(raw_season) == expected
