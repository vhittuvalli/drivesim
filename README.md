# DriveSim

Browser-based self-driving simulator (three.js / WebGL, no build step).

## Run

    npm run dev        # static server on :8000 with caching disabled (scripts/serve.py)
    open http://localhost:8000

URL params: `?seed=<int>` (city layout), `?hour=4..22` (time of day), `?cam=chase|hood|orbit|top`,
`?cars=0..120` (moving traffic), `?peds=0..200` (pedestrians), `?vans=<n>` (double-parked delivery vans),
`?weather=clear|rain|fog|snow`, `?scenario=<id>` (see below), `?debug=1` (planner overlay).
The **Link** button copies a URL with the current setup.

Keys: `1`–`4` switch camera, `Space` pauses, `M` take the wheel / hand back, `O` planner overlay,
`R` start / stop dataset recording.

### Driving yourself

Press `M` (or **Drive**) and steer with WASD / arrow keys or a gamepad (left stick, triggers). Press `M` again
to hand back to the expert; it needs you to be in a right-hand lane, heading along it. If you aren't, a
second `M` puts the car on the nearest lane. Contacts with vehicles or pedestrians are reported.

### Scenarios

Pick one from **Scenario** (or `?scenario=`); ↻ restarts it. Each places the ego on a straight block, scripts
the other actors, and passes or fails (any contact fails; so does running out of time).

| id | What happens |
|---|---|
| `overtake-parked` | A delivery van with hazards blocks the lane; pass it through the oncoming lane |
| `overtake-oncoming` | Same, with oncoming cars: wait for a gap, then pass |
| `overtake-slow` | A slow utility van; pass once the oncoming lane is clear and the pass ends before the next stop line |
| `lead-brake` | The car ahead brakes to a stop at 8 m/s² (then stalls, and gets overtaken) |
| `jaywalker` | A pedestrian runs out from in front of a parked van |
| `pull-out` | A parked car pulls into the lane just ahead |
| `red-runner` | You have a green light; a car on the cross street runs its red |
| `unprotected-left` | Left turn on green through a stream of oncoming traffic |

### Planner overlay (`O`)

Cyan ribbon: planned path (it bends into the oncoming lane while overtaking), dot: pure-pursuit target.
Red box: the vehicle being followed or yielded to; orange box and dot: a vehicle predicted to cut across
the path, and where. Rings: pedestrians on the road (red = the one being yielded to). Bar: next stop
line in the signal's color. Lane patch: the oncoming-lane check for an overtake (green clear, red
blocked, blue passing).

### Weather

Rain, fog and snow change the look (overcast sky, precipitation, wet or snowy ground) and the driving:
less grip means longer headways, gentler braking targets and slower corners for every driver, and fog
caps speed so the car can stop within the visible distance.

### Dataset recording (`R`)

Renders the roof camera at 10 Hz (320×160) and, on stop, downloads a zip with `frames/NNNNNN.jpg`,
`labels.csv` (steer, throttle, speed, acceleration, pose, expert or manual, the planner's reason, next
turn, overtake state, weather, hour) and `meta.json`. Recording while you drive gives human
demonstrations; recording the expert gives privileged-driver labels.

## Test

    npm test                       # headless simulation tests (Node 22+, no dependencies)
    npm run fuzz -- 40 3 rain      # seed sweep: seeds, minutes each, weather

The suite drives the ego expert, NPC traffic and pedestrians for several simulated minutes and
asserts no collisions, no red-light runs, and pedestrians staying on sidewalks unless crossing; it
also plays every scenario on several seeds. The fuzzer runs the same checks (with double-parked vans)
over many seeds and prints the seed of any failure so it can be reproduced with `?seed=`.
CI runs the tests and a short fuzz on every push to `main` and every pull request.

## Status

- **Phase 1: 3D city (done).** Procedural street grid with signalized intersections, buildings, street furniture, day/night lighting, and an expert driver that follows a random route.
- **Phase 2: traffic and pedestrians (done).** NPC cars with IDM car-following, signal compliance,
  left-turn gap acceptance and don't-block-the-box; parked cars; pedestrians on sidewalks and
  crosswalks with occasional jaywalking; the ego yields to all of them.
- **Scenarios and tools (done).** Overtaking (ego and NPCs), crossing-vehicle prediction, eight scripted
  scenarios, manual driving, planner overlay, weather, dataset recording, fuzzing.
- Phase 3: realistic sensor model (noise, blur, exposure, latency).
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
| `src/traffic.js` | Parked cars, NPC traffic and scripted vehicles (double-parked vans, scenario actors) |
| `src/peds.js` | Pedestrian simulation (sidewalks, crosswalks, jaywalking) |
| `src/pedRender.js` | Animated pedestrian rendering |
| `src/vehicle.js` | Bicycle-model physics + sedan model with sensor rig |
| `src/planner.js` | Routes and shared driving logic (curve speed, signals, yielding, IDM, crossing prediction, overtaking) + ego expert |
| `src/sim.js` | Headless world: stepping, manual/expert control, contacts, weather conditions, scenario runner |
| `src/scenarios.js` | Scripted scenarios and their pass/fail checks |
| `src/input.js` | Keyboard / gamepad driving input |
| `src/debug.js` | Planner overlay |
| `src/weather.js` | Weather look and rain/snow particles |
| `src/recorder.js`, `src/zip.js` | Camera dataset recorder and a minimal ZIP writer |
| `src/main.js` | Renderer, sky/sun, post-processing, cameras, HUD, controls |

## Assets

Textures in `assets/tex/` are from [Poly Haven](https://polyhaven.com) (CC0): asphalt_02, concrete_pavement,
concrete_floor_worn_02, brick_wall_02, concrete_panels, clay_plaster, bark_brown_02.
The pedestrian model (Xbot) is loaded at runtime from the three.js examples via jsDelivr.
