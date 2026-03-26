from utils.Event import Event
from typing import TypeAlias

mm:     TypeAlias = float
degree: TypeAlias = float
radian: TypeAlias = float
Point:  TypeAlias = tuple[mm, mm]
second: TypeAlias = float

__all__ = ["Event", "mm", "degree", "radian", "Point", "second"]
