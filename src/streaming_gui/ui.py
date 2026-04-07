from .ui_layout import StreamGUILayoutMixin
from .ui_phase_widgets import StreamGUIPhaseWidgetsMixin
from .ui_slot_widgets import StreamGUISlotWidgetsMixin


class StreamGUIUIMixin(StreamGUILayoutMixin, StreamGUISlotWidgetsMixin, StreamGUIPhaseWidgetsMixin):
    """Composed UI mixin split by layout, slot widget, and phase control concerns."""
