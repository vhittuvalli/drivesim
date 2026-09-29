# DriveSim

Browser-based self-driving simulator (three.js / WebGL, no build step).

## Run

    npm run dev        # python3 -m http.server 8000
    open http://localhost:8000

URL params: `?seed=<int>` (city layout), `?hour=4..22` (time of day), `?cam=chase|hood|orbit|top`,
`?cars=0..120` (moving traffic), `?peds=0..200` (pedestrians).
Keys: `1`–`4` switch camera, `Space` pauses.

## Test

    npm test           # headless simulation tests (Node 22+, no dependencies)

The suite drives the ego expert, NPC traffic and pedestrians for several simulated minutes and
asserts no collisions, no red-light runs, and pedestrians staying on sidewalks unless crossing.
CI runs it on every push to `main` and every pull request.

## Status

- **Phase 1: 3D city (done).** Procedural street grid with signalized intersections, buildings, street furniture, day/night lighting, and an expert driver that follows a random route.
- **Phase 2: traffic and pedestrians (done).** NPC cars with IDM car-following, signal compliance,
  left-turn gap acceptance and don't-block-the-box; parked cars; pedestrians on sidewalks and
  crosswalks with occasional jaywalking; the ego yields to all of them.
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
| `src/bodytypes.js` | Vehicle body dimensions and paint distribution |
| `src/fleet.js` | Instanced rendering of all non-ego vehicles |
| `src/traffic.js` | Parked cars and NPC traffic simulation |
| `src/peds.js` | Pedestrian simulation (sidewalks, crosswalks, jaywalking) |
| `src/pedRender.js` | Animated pedestrian rendering |
| `src/vehicle.js` | Bicycle-model physics + sedan model with sensor rig |
| `src/planner.js` | Routes and shared driving logic (curve speed, signals, yielding, IDM) + ego expert |
| `src/main.js` | Renderer, sky/sun, post-processing, cameras, HUD |

## Assets

Textures in `assets/tex/` are from [Poly Haven](https://polyhaven.com) (CC0): asphalt_02, concrete_pavement,
concrete_floor_worn_02, brick_wall_02, concrete_panels, clay_plaster, bark_brown_02.
The pedestrian model (Xbot) is loaded at runtime from the three.js examples via jsDelivr.
