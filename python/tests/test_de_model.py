import sympy as sp
from amici.de_model_components import Event
from amici.import_utils import amici_time_symbol
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_event_trigger_time():
    e = Event(
        sp.Symbol("event1"), "event name", amici_time_symbol - 10, sp.Float(0)
    )
    assert e.triggers_at_fixed_timepoint() is True
    assert e.get_trigger_time() == 10

    # fixed, but multiple timepoints - not (yet) supported
    e = Event(
        sp.Symbol("event1"),
        "event name",
        sp.sin(amici_time_symbol),
        sp.Float(0),
    )
    assert e.triggers_at_fixed_timepoint() is False

    e = Event(
        sp.Symbol("event1"), "event name", amici_time_symbol / 2, sp.Float(0)
    )
    assert e.triggers_at_fixed_timepoint() is True
    assert e.get_trigger_time() == 0

    # parameter-dependent triggers - not (yet) supported
    e = Event(
        sp.Symbol("event1"),
        "event name",
        amici_time_symbol - sp.Symbol("delay"),
        sp.Float(0),
    )
    assert e.triggers_at_fixed_timepoint() is False
