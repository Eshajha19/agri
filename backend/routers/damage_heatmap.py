"""Crop Damage Heatmap — returns grid-based damage severity data for field visualization."""

import logging
import random
import zlib
from typing import List, Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

GRID_ROWS = 10
GRID_COLS = 10

SEVERITY_LABELS = {0: "None", 1: "Minor", 2: "Moderate", 3: "Severe", 4: "Critical"}


class HeatmapCell(BaseModel):
    row: int
    col: int
    severity: int
    label: str
    damage_pct: float


class FieldSummary(BaseModel):
    total_cells: int
    damaged_cells: int
    avg_damage_pct: float
    yield_loss_projection_pct: float
    estimated_yield_loss_qtl: float
    estimated_area_affected_acres: float


class HeatmapResponse(BaseModel):
    grid: List[List[HeatmapCell]]
    summary: FieldSummary
    crop_type: str
    field_size_acres: float


@router.get("/damage-heatmap/{field_id}", response_model=HeatmapResponse)
async def get_damage_heatmap(field_id: str):
    random.seed(zlib.adler32(field_id.encode()))

    crop_type = random.choice(["Paddy", "Wheat", "Maize", "Cotton", "Sugarcane"])
    field_size = round(random.uniform(1.0, 10.0), 2)

    grid: List[List[HeatmapCell]] = []
    damaged = 0
    total_damage = 0.0

    for r in range(GRID_ROWS):
        row: List[HeatmapCell] = []
        for c in range(GRID_COLS):
            sev = random.choices([0, 1, 2, 3, 4], weights=[30, 25, 20, 15, 10])[0]
            pct_map = {0: 0.0, 1: random.uniform(5, 20), 2: random.uniform(21, 50), 3: random.uniform(51, 75), 4: random.uniform(76, 100)}
            damage_pct = round(pct_map[sev], 1)
            if sev > 0:
                damaged += 1
                total_damage += damage_pct
            row.append(HeatmapCell(row=r, col=c, severity=sev, label=SEVERITY_LABELS[sev], damage_pct=damage_pct))
        grid.append(row)

    avg_damage = round(total_damage / damaged, 1) if damaged > 0 else 0.0

    affected_ratio = damaged / (GRID_ROWS * GRID_COLS)
    yield_loss_pct = round(avg_damage * 0.65, 1)
    base_yield_per_acre = random.choice([18, 22, 25, 30, 35])
    estimated_yield_loss = round(field_size * affected_ratio * base_yield_per_acre * (avg_damage / 100) * 0.65, 2)
    affected_acres = round(field_size * affected_ratio, 2)

    return HeatmapResponse(
        grid=grid,
        summary=FieldSummary(
            total_cells=GRID_ROWS * GRID_COLS,
            damaged_cells=damaged,
            avg_damage_pct=avg_damage,
            yield_loss_projection_pct=yield_loss_pct,
            estimated_yield_loss_qtl=estimated_yield_loss,
            estimated_area_affected_acres=affected_acres,
        ),
        crop_type=crop_type,
        field_size_acres=field_size,
    )
