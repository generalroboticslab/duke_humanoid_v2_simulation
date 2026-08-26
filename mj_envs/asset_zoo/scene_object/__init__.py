"""Scene-object asset library: things the robot works around / with in a manipulation scene.

Procedural assets (the LARBANKE workbench, a shelf) built from shared primitives in ``base.py``
(``Prop`` + ``PartBuilder`` + ``add_props_to_spec``). Will grow to more furniture, rigid graspable
objects, and articulated objects (drawers/doors — those get MJCF + meshes under ``asset/`` and a
loader here, mirroring the robot-entity pattern, and use their own builder, not ``PartBuilder``).

Re-exports the builders + shared primitives so callers do
``from asset_zoo.scene_object import workbench, shelf, Prop, add_props_to_spec``.
"""

from asset_zoo.scene_object.base import (
    INVIS,
    PartBuilder,
    Prop,
    add_props_to_spec,
)
from asset_zoo.scene_object.human_figure import (
    DEFAULT_HUMAN,
    SHIRT_RGBA,
    HumanSpec,
    human_figure,
)
from asset_zoo.scene_object.human_hand import (
    DEFAULT_HAND,
    HAND_RGBA,
    HandSpec,
    human_hand,
)
from asset_zoo.scene_object.shelf import (
    DEFAULT_SHELF,
    SHELF_RGBA,
    ShelfSpec,
    shelf,
    tier_surface_z,
)
from asset_zoo.scene_object.workbench import (
    DEFAULT_TABLE_Z,
    LARBANKE,
    TOP_DEPTH_H,
    WorkbenchSpec,
    workbench,
)

__all__ = [
    "INVIS",
    "PartBuilder",
    "Prop",
    "add_props_to_spec",
    "DEFAULT_HUMAN",
    "SHIRT_RGBA",
    "HumanSpec",
    "human_figure",
    "DEFAULT_HAND",
    "HAND_RGBA",
    "HandSpec",
    "human_hand",
    "DEFAULT_SHELF",
    "SHELF_RGBA",
    "ShelfSpec",
    "shelf",
    "tier_surface_z",
    "DEFAULT_TABLE_Z",
    "LARBANKE",
    "TOP_DEPTH_H",
    "WorkbenchSpec",
    "workbench",
]
