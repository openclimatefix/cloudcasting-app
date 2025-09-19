import pandas as pd
import pytest
from tests.utils import make_sat_data


@pytest.fixture()
def init_time():
    return pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("30min")


@pytest.fixture()
def sat_5_data(init_time):
    times = pd.date_range(
        init_time - pd.Timedelta("3h"),
        init_time,
        freq=f"5min",
    )
    return make_sat_data(times)