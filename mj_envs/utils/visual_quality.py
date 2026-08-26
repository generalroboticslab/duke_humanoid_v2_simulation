"""Post-compile visual upgrades for the play-time MuJoCo scene.

Why this exists
---------------
mjlab scenes are lit for throughput, not for looking at. The compiled Argus scene ships
two lights that both point straight down (``dir = (0, 0, -1)``) plus MuJoCo's camera-attached
headlight with ``specular = 0``. Straight-down light plus a camera-coincident fill produces
near-zero shading gradient across a curved surface, so every part of the robot reads as the
same flat patch of its base color and the viewer loses all sense of form and ground contact.

The fix is standard three-point lighting, applied to the lights that already exist. Nothing
here changes physics: ``light_*``, ``mat_*`` and ``vis.*`` are render-only fields.

Design decisions
----------------
* **Re-aim, don't add.** ``nlight`` is fixed at compile time, so new lights would mean forking
  the mjlab spec builder for every task. Re-aiming the two existing lights costs nothing and
  works for any scene that has at least one.
* **One shadow caster.** Only the key light casts. Two shadow casters give a robot two
  contradictory ground shadows, which reads as *less* physical, not more.
* **Headlight demoted to fill.** Killing it outright leaves the shadow side pure black
  (MuJoCo has no bounce light or ambient occlusion). Keeping it weak with nonzero specular
  gives highlights that track the eye, which is what sells curvature on a metal part.
* **Rejected: skybox + haze.** ``mjVIS_HAZE`` needs a skybox texture and textures cannot be
  added after compile. That belongs in the scene XML, not here.

Known ceiling
-------------
MuJoCo's renderer is fixed-function Phong: no ambient occlusion, no global illumination, no
physically-based materials. This makes the scene *legible* (form, depth, contact), not
photoreal. Photoreal needs an external renderer.
"""

import numpy as np

# Key light aimed down/front/side rather than straight down. Elevation ~55 deg below
# horizontal: high enough that shadows stay short and the robot does not occlude itself,
# shallow enough to still rake across the top surfaces and reveal their curvature.
_KEY_DIR = np.array([0.4, 0.6, -1.0])
# Fill opposes the key in azimuth and sits shallower, to lift the shadow side without
# flattening the gradient the key just created.
_FILL_DIR = np.array([-0.5, -0.4, -1.0])
_FILL_SCALE = 0.35

# --- Planetary lighting (see apply_planetary_lighting) ---------------------------------------
# Sun elevation above the horizon. Straight overhead (mjlab's default) puts zero shading gradient
# on a horizontal surface, which erases exactly the crater and rock relief these bakes exist for.
# Apollo landing sites were deliberately chosen at low sun for the same reason; 30 deg keeps
# shadows readable without the half-frame darkness of a 10 deg terminator shot.
_SUN_ELEVATION_DEG = 30.0
_SUN_AZIMUTH_DEG = 135.0
# Solar irradiance relative to 1 AU. The Moon shares Earth's orbit, so its sun is NOT dimmer --
# what makes a lunar frame read as lunar is the total absence of skylight, not a weak sun. Mars
# at 1.524 AU gets 1/1.524^2 of it, which is the one place a real intensity difference belongs.
_MARS_IRRADIANCE = 1.0 / 1.524**2  # 0.431
# Peak MuJoCo diffuse for the single sun. mjlab ships 0.7 on *two* coincident overhead lights, so
# the scene was lit at 1.4 from straight up; one sun at 0.9 is both dimmer and directional.
_SUN_DIFFUSE = 0.9
# Sun colour. Airless Moon gets the unfiltered solar spectrum; Mars dust scatters out the blue,
# which is why MSL white-balance targets come back visibly warm.
_MOON_SUN_RGB = np.array([1.0, 1.0, 1.0])
_MARS_SUN_RGB = np.array([1.0, 0.87, 0.72])
# Earth sun: 1 AU baseline, near-white with slight warm tint (CIE D65 ~5750K vs 5778K solar).
_EARTH_SUN_RGB = np.array([1.0, 0.97, 0.92])
# Ambient fill, i.e. everything that is not the sun. The Moon has no atmosphere, so its only fill
# is regolith bounce -- near zero, and neutral. Mars' dust loading scatters a large, strongly
# reddened skylight component; this is why lunar shadows are black and martian ones are not.
# Earth's Rayleigh scattering gives a strong cool-blue skylight fill.
_MOON_AMBIENT = np.array([0.015, 0.015, 0.016])
_MARS_AMBIENT = np.array([0.100, 0.065, 0.045])
_EARTH_AMBIENT = np.array([0.090, 0.110, 0.150])
# Sky radiance, carried on `vis.rgba.haze` (see apply_planetary_lighting for why that field).
# Magnitude is baked into the RGB rather than split into a separate strength, so there is exactly
# one number per channel to tune. Moon: black -- at an exposure that holds sunlit regolith, stars
# sit ~10 stops down, which is why Apollo surface photography has empty black skies. Mars: the
# butterscotch of a dust-loaded sky, at ~0.35 radiance so it contributes roughly a third of the
# ground illumination (about right for the tau~0.5 dust opacity these bakes imply).
# Authored intent for Mars was post-tonemap (0.55, 0.45, 0.35) -- pale tan-orange, matching
# Path/Forward hazecam midday sky at tau~1. AgX desaturates low-magnitude colours so hard that
# source (0.35, 0.25, 0.17) lands at (0.61, 0.54, 0.47) -- warm pink, not butterscotch. These are
# the pre-compensated source values that produce the intended output under EEVEE AgX +
# EXPOSURE_STOPS=0.3, measured on grl1 (Blender 5.2 LTS):
# (0.160, 0.115, 0.070) -> post-AgX (0.478, 0.404, 0.314), within 13% of target on all channels.
_MOON_SKY = np.array([0.0, 0.0, 0.0])
_MARS_SKY = np.array([0.160, 0.115, 0.070])
# Earth sky: brighter than Mars (more atmosphere) and bluer (Rayleigh scattering favours short
# wavelengths). Same AgX pre-compensation logic as the Mars entry above.
_EARTH_SKY = np.array([0.110, 0.140, 0.190])

# Per-body dispatch for `apply_planetary_lighting`. The constants above are grouped by property
# so the bodies can be read against each other (every sun colour together, every ambient
# together); this table is the one place they are grouped by body, so adding one is a row rather
# than an edit inside four parallel conditionals. Irradiance is 1.0 for anything sharing Earth's
# orbit -- only Mars is genuinely further out.
_BODY_LIGHTING = {
  #        sun colour       irradiance         ambient         sky
  "moon": (_MOON_SUN_RGB, 1.0, _MOON_AMBIENT, _MOON_SKY),
  "mars": (_MARS_SUN_RGB, _MARS_IRRADIANCE, _MARS_AMBIENT, _MARS_SKY),
  "earth": (_EARTH_SUN_RGB, 1.0, _EARTH_AMBIENT, _EARTH_SKY),
}


def _sun_direction() -> np.ndarray:
  """Unit vector a ray of sunlight travels along, from `_SUN_ELEVATION_DEG` / `_SUN_AZIMUTH_DEG`.

  MuJoCo's ``light_dir`` is the propagation direction, so it points *away* from the sun's position
  in the sky -- hence the negation.
  """
  el, az = np.radians(_SUN_ELEVATION_DEG), np.radians(_SUN_AZIMUTH_DEG)
  return -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def apply_planetary_lighting(model, body: str) -> None:
  """Relight ``model`` in place as one of `_BODY_LIGHTING`. Render-only; does not touch dynamics.

  Why this is on the MuJoCo model and not in the Blender viewer
  ------------------------------------------------------------
  ``photoreal/bridge.py`` already ships every light's type, direction, colour and the headlight
  ambient, and ``blender_view._world_and_lights`` rebuilds the rig from them precisely so the two
  renderers cannot disagree. Tuning a sun inside Blender would fork that. Setting it here fixes
  the native viewer and the photoreal one from one place.

  What it changes
  ---------------
  * **One sun, aimed.** mjlab gives every scene two lights both pointing straight down: a
    directional and a tracking spot. Two coincident overhead sources is twice the irradiance from
    the worst possible angle. The directional lights become the sun at `_SUN_ELEVATION_DEG`;
    everything else is switched off, because no body here has a second light source.
  * **Per-body irradiance and colour**, from `_BODY_LIGHTING`.
  * **Ambient** set to the body's actual skylight, which is what separates a lunar frame from a
    martian or terrestrial one far more than the sun does.
  * **Sky colour** on ``vis.rgba.haze``. MuJoCo has no background-colour field -- haze is its
    atmosphere colour, is meaningless without a skybox (no scene here has one), and is left at
    white everywhere in this repo, so it is free to carry this. The Blender side treats a pure
    white haze as "unset" and keeps its old ambient-derived background, so no other scene moves.

  Args:
      model: compiled ``mujoco.MjModel``.
      body: a key of `_BODY_LIGHTING` (``"moon"``, ``"mars"``, ``"earth"``).
  """
  assert body in _BODY_LIGHTING, f"unknown body {body!r}"
  rgb, irradiance, ambient, sky = _BODY_LIGHTING[body]
  peak = _SUN_DIFFUSE * irradiance

  # mjtLightType.mjLIGHT_DIRECTIONAL == 1. Matched by type rather than index because the light
  # order depends on which entity's spec was merged first.
  is_sun = model.light_type == 1
  model.light_dir[is_sun] = _sun_direction()
  model.light_diffuse[is_sun] = rgb * peak
  model.light_specular[is_sun] = rgb * peak * 0.1  # regolith is matte; a tight hot spot reads wet
  model.light_active[~is_sun] = 0

  hl = model.vis.headlight
  hl.ambient[:] = ambient
  hl.diffuse[:] = 0.0  # a camera-attached lamp is the one light no body has
  hl.specular[:] = 0.0
  model.vis.rgba.haze[:3] = sky


def apply_realistic_visuals(model) -> None:
    """Relight ``model`` in place for viewing. Render-only; does not touch dynamics.

    Args:
        model: compiled ``mujoco.MjModel``. Needs ``nlight >= 1`` to do anything useful;
            with zero lights only the headlight and material tweaks apply.
    """
    if model.nlight:
        key_dir = _KEY_DIR / np.linalg.norm(_KEY_DIR)
        model.light_dir[0] = key_dir
        model.light_diffuse[0] = (0.85, 0.83, 0.78)  # slightly warm key
        model.light_specular[0] = (0.35, 0.35, 0.35)
        model.light_castshadow[0] = 1

        if model.nlight > 1:
            fill_dir = _FILL_DIR / np.linalg.norm(_FILL_DIR)
            model.light_dir[1:] = fill_dir
            model.light_diffuse[1:] = np.array((0.78, 0.82, 0.9)) * _FILL_SCALE  # cool fill
            model.light_specular[1:] = 0.0
            model.light_castshadow[1:] = 0  # single shadow caster; see module docstring

    hl = model.vis.headlight
    hl.ambient[:] = (0.18, 0.18, 0.2)
    hl.diffuse[:] = (0.2, 0.2, 0.2)
    hl.specular[:] = (0.25, 0.25, 0.25)

    # Shadow frustum is sized as shadowclip * model extent. The default 1.0 spans the whole
    # terrain grid, so the 8192 shadow map lands only a few texels on the robot and its
    # shadow degrades to mush. Clamp the frustum to the robot's neighbourhood instead.
    model.vis.map.shadowclip = 0.1
    model.vis.quality.shadowsize = 8192
    model.vis.quality.offsamples = 8  # 4 -> 8: kills stair-stepping on the thin link geoms

    # Metal parts need a tight, bright highlight to read as metal; the mjlab default 0.5/0.5
    # reads as matte plastic. Ground keeps a low reflectance so the robot is visually anchored.
    model.mat_specular[:] = 0.6
    model.mat_shininess[:] = 0.7


def demo() -> None:
    """Self-check: relighting must change render state and leave dynamics untouched."""
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <light pos="0 0 2" dir="0 0 -1"/>
          <light pos="0 0 3" dir="0 0 -1"/>
          <body><freejoint/><geom size=".1" mass="1"/></body>
        </worldbody>
      </mujoco>
    """)
    before_mass = model.body_mass.copy()
    apply_realistic_visuals(model)

    assert not np.allclose(model.light_dir[0], (0, 0, -1)), "key light not re-aimed"
    assert model.light_castshadow[0] == 1, "key must cast"
    assert model.light_castshadow[1] == 0, "fill must not cast"
    assert model.vis.headlight.specular[0] > 0, "headlight needs specular for highlights"
    assert np.array_equal(model.body_mass, before_mass), "dynamics must be untouched"

    # Planetary relight: one aimed sun, the rest off, and Mars strictly dimmer / warmer / less
    # contrasty than the Moon. Built with mjlab's actual rig -- a directional plus a spot, both
    # straight down -- because "kill the second light" is the half of this that a one-light
    # fixture would not exercise.
    lit = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <light type="directional" pos="0 0 2" dir="0 0 -1"/>
          <light type="spot" pos="0 0 3" dir="0 0 -1"/>
          <body><freejoint/><geom size=".1" mass="1"/></body>
        </worldbody>
      </mujoco>
    """)
    lit_mass = lit.body_mass.copy()
    apply_planetary_lighting(lit, "moon")
    assert lit.light_active[0] == 1 and lit.light_active[1] == 0, "sun on, everything else off"
    assert lit.light_dir[0][2] < 0 and abs(lit.light_dir[0][0]) > 0.1, "sun must rake, not point down"
    assert np.allclose(np.linalg.norm(lit.light_dir[0]), 1.0), "sun direction must be unit"
    assert np.allclose(lit.vis.rgba.haze[:3], 0.0), "lunar sky is black"
    moon_sun, moon_amb = lit.light_diffuse[0].copy(), lit.vis.headlight.ambient.copy()
    assert np.array_equal(lit.body_mass, lit_mass), "dynamics must be untouched"

    apply_planetary_lighting(lit, "mars")
    assert lit.light_diffuse[0].max() < moon_sun.max(), "Mars is 1.5 AU out; its sun is dimmer"
    assert lit.light_diffuse[0][0] > lit.light_diffuse[0][2], "Mars sunlight is reddened by dust"
    assert lit.vis.headlight.ambient[0] > moon_amb[0], "Mars has skylight; the Moon has none"
    assert lit.vis.rgba.haze[0] > lit.vis.rgba.haze[2] > 0.0, "Mars sky is butterscotch, not black"

    # Earth's discriminator is the opposite of Mars': Rayleigh scattering makes both its skylight
    # and its sky blue-dominant, where dust makes Mars' red-dominant. Catches a table row wired to
    # the wrong body, which the shared irradiance of 1.0 would otherwise hide.
    apply_planetary_lighting(lit, "earth")
    assert np.allclose(lit.light_diffuse[0].max(), moon_sun.max()), "Earth shares the Moon's orbit"
    assert lit.vis.headlight.ambient[2] > lit.vis.headlight.ambient[0], "Earth skylight is blue"
    assert lit.vis.rgba.haze[2] > lit.vis.rgba.haze[0] > 0.0, "Earth sky is blue, not black"
    print("visual_quality demo ok")


if __name__ == "__main__":
    demo()
