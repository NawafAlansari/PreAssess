"""
Seattle GIS context: which zoning, overlay districts, environmentally critical
areas, and street trees apply at a point.

All queries go to fixed City of Seattle ArcGIS feature services (the city's
authoritative layers). The overlay views all expose the same master layer
(DPD.ZONING_OVERLAY_ALL, layer 23) filtered by overlay family, and each
feature carries the SMC chapter that regulates it — which is what lets the
report cite the exact chapter instead of asking the resident to "confirm
whether the property is in an overlay district".
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional

import httpx

SEATTLE_GIS = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/ArcGIS/rest/services"

ZONING_QUERY = f"{SEATTLE_GIS}/Current_Land_Use_Zoning_Detail_2/FeatureServer/0/query"

OVERLAY_VIEWS = {
    "station_area": f"{SEATTLE_GIS}/Station_Area_Overlay_(light_rail)/FeatureServer/23/query",
    "historic_special_review": f"{SEATTLE_GIS}/Zoning_Overlays-Historic-Special_Review_Districts/FeatureServer/23/query",
    "major_institution": f"{SEATTLE_GIS}/Major_Institution_Overlay/FeatureServer/23/query",
    "shoreline": f"{SEATTLE_GIS}/Shoreline_Environments/FeatureServer/23/query",
    "additional": f"{SEATTLE_GIS}/Additional_Overlay_Areas/FeatureServer/23/query",
}

ECA_BASE = f"{SEATTLE_GIS}/Environmentally_Critical_Areas_ECA/FeatureServer"
ECA_LAYERS = {
    "flood_prone": 0,
    "known_slide": 1,
    "liquefaction_prone": 5,
    "peat_settlement": 6,
    "potential_slide": 7,
    "riparian_corridor": 8,
    "steep_slope": 9,
    "wetland": 10,
    "wildlife_habitat": 11,
}

TREES_QUERY = f"{SEATTLE_GIS}/Combined_Tree_Point/FeatureServer/0/query"


def point_params(lat: float, lon: float) -> Dict[str, str]:
    return {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "false",
        "f": "json",
    }


def chapter_prefix(chapter: Optional[str]) -> Optional[str]:
    """'Chapter 23.66.100' -> '23.66.100'; None if unparseable."""
    if not chapter:
        return None
    match = re.search(r"(\d{2}\.\d+[A-Za-z]?(?:\.[0-9A-Za-z]+)*)", chapter)
    return match.group(1) if match else None


async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, str]) -> dict:
    resp = await client.get(url, params=params, timeout=12.0)
    resp.raise_for_status()
    return resp.json()


async def _zoning(client: httpx.AsyncClient, lat: float, lon: float) -> Optional[dict]:
    params = point_params(lat, lon) | {"outFields": "ZONING,ORDINANCE,PEDESTRIAN,SHORELINE"}
    body = await _get_json(client, ZONING_QUERY, params)
    features = body.get("features", [])
    if not features:
        return None
    attrs = features[0]["attributes"]
    return {
        "zone": (attrs.get("ZONING") or "").strip() or None,
        "ordinance": (attrs.get("ORDINANCE") or "").strip() or None,
        "pedestrian": (attrs.get("PEDESTRIAN") or "").strip() or None,
    }


async def _overlays(client: httpx.AsyncClient, lat: float, lon: float) -> List[dict]:
    params = point_params(lat, lon) | {
        "outFields": "OVERLAY,DESCRIPTION,PUBLIC_DESCRIPTION,TYPE,CHAPTER,CHAPTER_LINK"
    }
    results = await asyncio.gather(
        *(_get_json(client, url, params) for url in OVERLAY_VIEWS.values()),
        return_exceptions=True,
    )
    overlays: List[dict] = []
    seen = set()
    for family, body in zip(OVERLAY_VIEWS.keys(), results):
        if isinstance(body, BaseException):
            continue
        for feature in body.get("features", []):
            attrs = feature["attributes"]
            key = (attrs.get("TYPE"), attrs.get("DESCRIPTION"))
            if key in seen:
                continue
            seen.add(key)
            chapter = (attrs.get("CHAPTER") or "").strip() or None
            overlays.append(
                {
                    "family": family,
                    "name": (attrs.get("DESCRIPTION") or "").strip() or None,
                    "type": (attrs.get("TYPE") or "").strip() or None,
                    "code": (attrs.get("OVERLAY") or "").strip() or None,
                    "about": (attrs.get("PUBLIC_DESCRIPTION") or "").strip() or None,
                    "chapter": chapter,
                    "chapter_prefix": chapter_prefix(chapter),
                    "chapter_link": (attrs.get("CHAPTER_LINK") or "").strip() or None,
                }
            )
    return overlays


async def _eca_flags(client: httpx.AsyncClient, lat: float, lon: float) -> List[str]:
    params = point_params(lat, lon) | {"returnCountOnly": "true"}

    async def count(layer_id: int) -> int:
        body = await _get_json(client, f"{ECA_BASE}/{layer_id}/query", params)
        return int(body.get("count", 0))

    results = await asyncio.gather(
        *(count(layer_id) for layer_id in ECA_LAYERS.values()), return_exceptions=True
    )
    return [
        name
        for name, result in zip(ECA_LAYERS.keys(), results)
        if isinstance(result, int) and result > 0
    ]


async def _trees(client: httpx.AsyncClient, lat: float, lon: float, radius_m: int) -> dict:
    params = point_params(lat, lon) | {
        "distance": str(radius_m),
        "units": "esriSRUnit_Meter",
        "outFields": "COMMON_NAME,SCIENTIFIC_NAME,DBH,OWNERSHIP,UNITDESC",
        "resultRecordCount": "200",
        "returnGeometry": "true",
        "outSR": "4326",
    }
    body = await _get_json(client, TREES_QUERY, params)
    trees = []
    for feature in body.get("features", []):
        attrs = feature["attributes"]
        geom = feature.get("geometry") or {}
        trees.append(
            {
                "common_name": attrs.get("COMMON_NAME"),
                "scientific_name": attrs.get("SCIENTIFIC_NAME"),
                "dbh_inches": attrs.get("DBH"),
                "ownership": attrs.get("OWNERSHIP"),
                "location": attrs.get("UNITDESC"),
                "lat": geom.get("y"),
                "lon": geom.get("x"),
            }
        )
    trees.sort(key=lambda t: t.get("dbh_inches") or 0, reverse=True)
    return {
        "radius_m": radius_m,
        "count": len(trees),
        "largest": trees[:8],
        "points": [
            {k: t[k] for k in ("lat", "lon", "common_name", "dbh_inches")}
            for t in trees
            if t.get("lat") is not None
        ][:200],
    }


async def point_context(lat: float, lon: float, tree_radius_m: int = 30) -> dict:
    """Everything the city's GIS knows about regulating this point."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        zoning, overlays, eca, trees = await asyncio.gather(
            _zoning(client, lat, lon),
            _overlays(client, lat, lon),
            _eca_flags(client, lat, lon),
            _trees(client, lat, lon, tree_radius_m),
            return_exceptions=True,
        )

    warnings: List[str] = []

    def ok(value, fallback, label):
        if isinstance(value, BaseException):
            warnings.append(f"{label} lookup failed: {value}")
            return fallback
        return value

    return {
        "zoning": ok(zoning, None, "zoning"),
        "overlays": ok(overlays, [], "overlays"),
        "eca": ok(eca, [], "eca"),
        "trees": ok(trees, {"radius_m": tree_radius_m, "count": 0, "largest": []}, "trees"),
        "warnings": warnings,
    }
