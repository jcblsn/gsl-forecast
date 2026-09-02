import numpy as np

from src.forecasting import hypsometry as h


def test_lookups_are_monotone_and_invertible():
    levels = np.array([4188.5, 4192.0, 4198.0, 4205.0])
    areas, volumes = h.area_km2(levels), h.volume_kaf(levels)
    assert np.all(np.diff(areas) > 0) and np.all(np.diff(volumes) > 0)
    assert 1500 < areas[1] < 1700
    assert np.allclose(h.elevation_ft(volumes), levels, atol=0.05)
