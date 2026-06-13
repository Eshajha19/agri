from datetime import date

import smart_farm_autopilot as autopilot_module


def test_build_sowing_schedule_uses_single_today_value(monkeypatch):
    class DateShim:
        calls = 0

        @classmethod
        def today(cls):
            cls.calls += 1
            if cls.calls == 1:
                return date(2024, 12, 31)
            return date(2025, 1, 1)

        def __new__(cls, *args, **kwargs):
            return date(*args, **kwargs)

    monkeypatch.setattr(autopilot_module, "date", DateShim)

    planner = autopilot_module.SmartFarmAutopilot()
    schedule = planner._build_sowing_schedule(
        "Wheat",
        {
            "sowing_month": 11,
            "germination_days": 7,
            "transplant_days": None,
            "duration_days": 120,
        },
        "Rabi",
    )

    assert DateShim.calls == 1
    assert schedule[0].sowing_start == "2024-11-01"
