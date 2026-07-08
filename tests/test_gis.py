import json

import httpx
import pytest

import api.gis as gis
from api.gis import chapter_prefix, point_context


@pytest.mark.parametrize(
    "chapter,expected",
    [
        ("Chapter 23.61", "23.61"),
        ("Chapter 23.66.100", "23.66.100"),
        ("SMC 23.60A.190", "23.60A.190"),
        ("", None),
        (None, None),
        ("no numbers here", None),
    ],
)
def test_chapter_prefix(chapter, expected):
    assert chapter_prefix(chapter) == expected


def fake_transport():
    """MockTransport emulating the Seattle GIS services."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "Current_Land_Use_Zoning_Detail_2" in url:
            body = {
                "features": [
                    {"attributes": {"ZONING": "NR2", "ORDINANCE": "126509", "PEDESTRIAN": " "}}
                ]
            }
        elif "Station_Area_Overlay" in url:
            body = {
                "features": [
                    {
                        "attributes": {
                            "OVERLAY": "ED",
                            "DESCRIPTION": "Columbia City",
                            "PUBLIC_DESCRIPTION": "TOD near light rail.",
                            "TYPE": "LIGHTRAIL",
                            "CHAPTER": "Chapter 23.61",
                            "CHAPTER_LINK": "https://example.test/23.61",
                        }
                    }
                ]
            }
        elif "Major_Institution_Overlay" in url:
            # Same feature repeated by another view: must be deduplicated.
            body = {
                "features": [
                    {
                        "attributes": {
                            "OVERLAY": "ED",
                            "DESCRIPTION": "Columbia City",
                            "TYPE": "LIGHTRAIL",
                            "CHAPTER": "Chapter 23.61",
                        }
                    }
                ]
            }
        elif "Environmentally_Critical_Areas_ECA/FeatureServer/9" in url:
            body = {"count": 2}  # steep slope
        elif "Environmentally_Critical_Areas_ECA" in url:
            body = {"count": 0}
        elif "Combined_Tree_Point" in url:
            body = {
                "features": [
                    {"attributes": {"COMMON_NAME": "Chinese Elm", "SCIENTIFIC_NAME": "Ulmus parvifolia", "DBH": 2, "OWNERSHIP": "SDOT", "UNITDESC": "319 6TH AVE N"}, "geometry": {"x": -122.29, "y": 47.56}},
                    {"attributes": {"COMMON_NAME": "Red Oak", "SCIENTIFIC_NAME": "Quercus rubra", "DBH": 24, "OWNERSHIP": "SDOT", "UNITDESC": "321 6TH AVE N"}, "geometry": {"x": -122.291, "y": 47.561}},
                ]
            }
        else:
            body = {"features": []}
        return httpx.Response(200, content=json.dumps(body))

    return httpx.MockTransport(handler)


@pytest.fixture()
def patched_gis_transport(monkeypatch):
    transport = fake_transport()
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(gis.httpx, "AsyncClient", client_factory)


@pytest.mark.anyio
async def test_point_context_shape(patched_gis_transport):
    ctx = await point_context(47.56, -122.29, tree_radius_m=30)

    assert ctx["zoning"]["zone"] == "NR2"

    assert len(ctx["overlays"]) == 1  # deduplicated across views
    overlay = ctx["overlays"][0]
    assert overlay["name"] == "Columbia City"
    assert overlay["chapter_prefix"] == "23.61"

    assert ctx["eca"] == ["steep_slope"]

    assert ctx["trees"]["count"] == 2
    assert ctx["trees"]["largest"][0]["common_name"] == "Red Oak"  # sorted by DBH
    assert ctx["trees"]["points"][0]["lat"] == 47.561
    assert ctx["warnings"] == []


@pytest.fixture()
def anyio_backend():
    return "asyncio"
