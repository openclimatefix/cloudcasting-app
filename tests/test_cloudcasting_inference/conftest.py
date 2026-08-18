import pandas as pd
import pytest

from tests.utils import make_sat_data, write_icechunk


@pytest.fixture()
def init_time():
    return pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("30min")


@pytest.fixture()
def sat_icechunk_path(tmp_path, init_time) -> str:
    # The model needs 165 minutes of history, and `SatelliteDataset` discards a run of timestamps
    # exactly as long as the sample it needs, so the app asks for one 15-minute step more than
    # that. 4 hours covers it with room to spare
    times = pd.date_range(
        init_time - pd.Timedelta("4h"),
        init_time,
        freq="5min",
    )
    return write_icechunk(make_sat_data(times), str(tmp_path / "sat.icechunk"))
