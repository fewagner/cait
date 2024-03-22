from abc import ABC, abstractmethod
from typing import List, Union, Callable, Tuple

class BackendBaseClass(ABC):

    @abstractmethod
    def __init__(template: str, height: int, width: int, show_controls: bool):
        ...

    @abstractmethod
    def _show_legend(show: bool = True):
        ...

    @abstractmethod
    def _add_line(x: List[float], y: List[float], name: str = None):
        ...

    @abstractmethod
    def _add_scatter(x: List[float], y: List[float], name: str = None):
        ...

    @abstractmethod
    def _add_histogram(bins: Union[int, tuple], data: List[float], name: str = None):
        ...

    @abstractmethod
    def _add_vmarker(marker_pos: List[float], y_int: Tuple[float], name: str = None):
        ...

    @abstractmethod
    def _update_line(name: str, x: List[float], y: List[float]):
        ...

    @abstractmethod
    def _update_scatter(name: str, x: List[float], y: List[float]):
        ...

    @abstractmethod
    def _update_histogram(name: str, bins: Union[int, tuple], data: List[float]):
        ...

    @abstractmethod
    def _update_vmarker(name: str, marker_pos: List[float], y_int: Tuple[float]):
        ...

    @abstractmethod
    def _set_axes(data: dict):
        ...

    @abstractmethod
    def _add_button(text: str, callback: Callable, tooltip: str = None, where: int = -1, key: str = None):
        ...

    @abstractmethod
    def _get_figure():
        ...

    @abstractmethod
    def _show():
        ...

    @abstractmethod
    def _update():
        ...

    @abstractmethod
    def _close(b=None):
        ...

    @property
    @abstractmethod
    def line_names(self):
        ...

    @property
    @abstractmethod
    def scatter_names(self):
        ...

    @property
    @abstractmethod
    def histogram_names(self):
        ...