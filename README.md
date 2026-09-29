# DriveSim

Browser-based self-driving simulator (three.js / WebGL, no build step).

## Run

    npm run dev        # python3 -m http.server 8000
    open http://localhost:8000

URL params: `?seed=<int>` (city layout), `?hour=4..22` (time of day), `?cam=chase|hood|orbit|top`.
Keys: `1`–`4` switch camera, `Space` pauses.

## Status

- **Phase 1: 3D city (done).** Procedural street grid with signalized intersections, buildings, street furniture, day/night lighting, and an expert driver that follows a random route.
- Phase 2: traffic and pedestrians.
- Phase 3: realistic sensor model (noise, blur, exposure, weather, latency).
- Phase 4: neural driving policy trained on sensor data.

The 2D prototype lives in `legacy/` (`/legacy/` on the dev server).

## Layout

| File | Role |
|---|---|
| `src/config.js` | City dimensions, vehicle constants, helpers |
| `src/materials.js` | PBR textures, procedural facades/storefronts, ground shader patch (anti-tiling, wear, night light pools) |
| `src/city.js` | Grid, sidewalks, markings, buildings, street lights, trees, signal hardware |
| `src/signals.js` | Fixed-time signal controller |
| `src/vehicle.js` | Bicycle-model physics + sedan model with sensor rig |
| `src/planner.js` | Route generation + expert driver (pure pursuit, speed profile, signal stops) |
| `src/main.js` | Renderer, sky/sun, post-processing, cameras, HUD |

## Assets

Textures in `assets/tex/` are from [Poly Haven](https://polyhaven.com) (CC0): asphalt_02, concrete_pavement,
concrete_floor_worn_02, brick_wall_02, concrete_panels, clay_plaster, bark_brown_02.
