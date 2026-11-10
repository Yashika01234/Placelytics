from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_renders_and_responds_to_filters():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    before = [m.value for m in at.markdown if "kpi" in m.value][:2]
    at.sidebar.multiselect[1].set_value(["CSE"]).run()   # branch filter
    assert not at.exception, at.exception
    after = [m.value for m in at.markdown if "kpi" in m.value][:2]
    assert before != after
    at.sidebar.multiselect[1].set_value([]).run()        # empty filter -> graceful error, no crash
    assert not at.exception and len(at.error) == 1
