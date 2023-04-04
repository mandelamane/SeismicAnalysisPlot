from pyqtgraph.dockarea import DockArea
from pyqtgraph.Qt import QtCore


class KeyPressWindow(DockArea):
    """
    A subclass of DockArea that emits a signal when a key is pressed.
    """

    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        """
        Call the parent class constructor.

        Parameters
        ----------
        *args : Tuple[Any]
            Positional arguments passed to the parent class constructor
        **kwargs : Dict[Any, Any]
            Keyword arguments passed to the parent class constructor
        """
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev: QtCore.QEvent):
        """
        Emit the signal with the key press event object.

        Parameters
        ----------
        ev : QtCore.QEvent
            The key press event object
        """
        self.sigKeyPress.emit(ev)
