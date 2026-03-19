from .ui_layout import StreamGUILayoutMixin
from .ui_slot_widgets import StreamGUISlotWidgetsMixin


class StreamGUIUIMixin(StreamGUILayoutMixin, StreamGUISlotWidgetsMixin):
    """Composed UI mixin split by layout and slot widget concerns."""
