import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class LinkageConfig:
    d: float
    base_y: float
    l1: float
    l2: float
    l3: float
    l4: float
    l5: float
    left_branch: int
    right_branch: int


@dataclass(frozen=True)
class SimulatorConfig:
    left_angle_deg: float
    right_angle_deg: float
    distance: float
    target_angle_deg: float
    closure_branch: int
    tip_mass_kg: float
    gravity_x: float
    gravity_y: float


@dataclass(frozen=True)
class SingularityPlotConfig:
    target_radius: float
    target_shape: str


DEFAULT_LINKAGE = LinkageConfig(
    d=0.173,
    base_y=-0.486,
    l1=0.556,
    l2=0.544,
    l3=0.544,
    l4=0.557,
    l5=0.406,
    left_branch=-1,
    right_branch=1,
)

DEFAULT_SIMULATOR = SimulatorConfig(
    left_angle_deg=15.534233824250094,
    right_angle_deg=43.922888743202634,
    distance=0.5,
    target_angle_deg=90.0,
    closure_branch=-1,
    tip_mass_kg=1.0,
    gravity_x=-9.81,
    gravity_y=0.0,
)

DEFAULT_SINGULARITY_PLOT = SingularityPlotConfig(
    target_radius=1.0,
    target_shape="circle",
)


CONFIG_SECTIONS = {
    "linkage": (DEFAULT_LINKAGE, LinkageConfig),
    "simulator": (DEFAULT_SIMULATOR, SimulatorConfig),
    "singularity_plot": (DEFAULT_SINGULARITY_PLOT, SingularityPlotConfig),
}


def _with_overrides(default, config_class, overrides):
    allowed = {field.name for field in fields(config_class)}
    unknown = set(overrides) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {config_class.__name__} option(s): {names}")
    return replace(default, **overrides)


def load_config_file(path=None):
    """Load an optional TOML config file, returning the three config objects."""
    if path is None:
        return DEFAULT_LINKAGE, DEFAULT_SIMULATOR, DEFAULT_SINGULARITY_PLOT

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    unknown_sections = set(raw_config) - set(CONFIG_SECTIONS)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unknown config section(s): {names}")

    configs = {}
    for section, (default, config_class) in CONFIG_SECTIONS.items():
        overrides = raw_config.get(section, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"config section [{section}] must be a table")
        configs[section] = _with_overrides(default, config_class, overrides)

    return configs["linkage"], configs["simulator"], configs["singularity_plot"]
