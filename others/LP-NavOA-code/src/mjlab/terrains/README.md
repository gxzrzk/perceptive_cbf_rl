# Terrains

Procedurally generate terrains using primitive geometries and heightfields.

Terrain generators are configured through `TerrainGeneratorCfg` and are used by
the LP-NavOA obstacle and wall-maze tasks. `ALL_TERRAINS_CFG` in `config.py`
collects the built-in presets, and
`uv run python src/mjlab/scripts/visualize_terrain.py` opens an interactive
viewer for inspecting them.
