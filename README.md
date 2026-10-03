# DriveSim

Browser-based self-driving simulator (three.js / WebGL, no build step).

## Run

    npm run dev        # static server on :8000 with caching disabled (scripts/serve.py)
    open http://localhost:8000

URL params: `?seed=<int>` (city layout), `?hour=4..22` (time of day), `?cam=chase|hood|orbit|top`,
`?cars=0..120` (moving traffic), `?peds=0..200` (pedestrians), `?vans=<n>` (double-parked delivery vans),
`?road=highway` (start on the highway), `?hwcars=0..160` (highway traffic),
`?weather=clear|rain|fog|snow`, `?scenario=<id>` (see below), `?debug=1` (planner overlay), `?speed=1|2|4|8|16`,
`?fx=0` (no post-processing), and for the learning tools `?collect=<frames>&noise=0..1`, `?neural=1`,
`?bench=1&trials=<n>` (see [Learning to drive](#learning-to-drive)).
The **Link** button copies a URL with the current setup.

Keys: `1`–`4` switch camera, `Space` pauses, `M` take the wheel / hand back, `O` planner overlay,
`R` start / stop dataset recording, `C` collect training data, `N` neural driver, `B` benchmark,
`G` city ⇄ highway.

### Highway

A divided motorway loops around the city (**Road** → Highway, or `G`): 5.2 km, three lanes each way,
a concrete median, guardrails, lamp posts, sign gantries, 300 m corners and a gentle S-bend on two
sides, at a 108 km/h limit. It's there to test drivers at speed: more distance covered per decision,
curves taken near the grip limit, and traffic that changes lanes around you.

Traffic on it follows IDM and decides lane changes with MOBIL (it changes when that improves its own
acceleration by more than a threshold plus a politeness-weighted cost to the cars behind, and the new
follower wouldn't have to brake hard). Drivers keep right except to pass, yield to cars merging ahead
of them, have their own cruising speeds (vans stand in for slower trucks), and move over for a stalled
car with its hazards on. The ego expert uses the same logic and looks up to 140 m ahead at speed.

### Driving yourself

Press `M` (or **Drive**) and steer with WASD / arrow keys or a gamepad (left stick, triggers). Press `M` again
to hand back to the expert; it needs you to be in a right-hand lane, heading along it (on the highway:
anywhere on your side of the road, heading along it; the expert merges into the nearest lane). If you
aren't, a second `M` puts the car on the nearest lane. Contacts with vehicles or pedestrians are reported.

### Scenarios

Pick one from **Scenario** (or `?scenario=`); ↻ restarts it. Each places the ego on a straight block (or a
random stretch of highway), scripts the other actors, and passes or fails (any contact fails; so does
running out of time).

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
| `hw-cut-in` | Highway, 100 km/h: a slower car swerves into your lane 14 m ahead and brakes; a car alongside blocks the other lane |
| `hw-stalled` | A broken-down car in the right lane; find a gap in the middle-lane traffic and move over |
| `hw-jam` | Traffic ahead brakes at 6.5 m/s² to a standstill in every lane, waits, then moves off |

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

## Learning to drive

A camera-based driving network is trained by imitation of the expert and runs in the browser.

**The network** (`train/model.py`) sees the 256×128 roof camera and the car's speed. For each navigation
command (left / straight / right at the next intersection) it predicts 8 waypoints and a target speed
(conditional imitation learning); the route's next turn picks the branch. The waypoints are 2–23 m
ahead up to 12 m/s and spread out in proportion to speed above that (57 m ahead at 30 m/s), so they
always cover about two seconds of driving. It also predicts coarse
segmentation and depth (auxiliary training targets) and an attention map. All branches are supervised
wherever the expert can label them, not just the one taken (`src/labels.js`). On the highway the same
three commands mean change lanes left / keep the lane / change lanes right: the command is the lane
change the expert is making, and while it keeps its lane the left and right branches are labeled with
the lane changes it could start right now.

**In the app** (`src/neural.js`) the network runs with ONNX Runtime Web (WASM). It observes at 10 Hz of
*simulation* time and the simulation waits for each answer, so it drives the same on a slow machine.
Between observations its path is held in world coordinates and followed with pure pursuit; the target
speed sets the throttle. The expert runs in shadow mode as a **safety driver** (`src/safety.js`): if the
network leaves its lane, points the wrong way, hasn't braked 0.3 s after the expert would brake hard, or
sits still for 2.5 s when the expert would pull away, the expert drives for 3 s and it counts as a
takeover. Training oversamples turns, lane changes, pulling away from a stop, hard braking and
frames with an obstacle close ahead.

**Traffic lights and obstacles.** A second, narrow camera (22°, pitched up) makes signal lamps a few
pixels across instead of one. Besides the paths and speeds, the network predicts the state of the
signal ahead, the distance to its stop line, and the gap to and speed of the obstacle on its path.
When it is confident the light is red (or amber with room to stop) the target speed is capped to stop
at the predicted line; with an obstacle output, the expert's car-following model (IDM) on the
predicted gap caps it too. Both caps only ever lower the network's own speed. The obstacle output
doesn't generalize well yet (gap error of ~±15-20 m for obstacles within 30 m on unseen roads), so
the committed model is trained without it.

**Neural** (`N`) shows the network's view: the camera frame with its attention map, its segmentation, a
live chart of its steering against the expert's (shaded where the safety driver drove) and the takeover
tally. The **Steering wheel** card shows the network's wheel next to the expert's as a driver would turn
them (15:1 steering ratio, ±516° lock to lock, moving at the car's steering rate), the angle between
them, each one's brake / throttle, and a glow on whichever is in control. With the planner overlay
(`O`) its predicted path is drawn in magenta (thin lines: the other command branches).

**Collect** (`C`, or `?collect=<frames>`) drives on its own and streams frames, label images and
`samples.jsonl` to `data/<run>/` through the dev server, one run per episode. Episodes randomize the road
(city or, 30% of the time, highway; `--highway 0.8` / `?hwshare=` to change that), weather, time of day,
traffic, pedestrians and double-parked vans, play a scripted scenario for that road about a third of the
time, and add correlated steering noise to half of the episodes so the data contains recoveries. On the
highway the expert picks a new preferred lane every 15–40 s, so the data covers every lane and plenty of
lane changes, not just cruising in the right lane. With the neural driver on, the network drives and the
expert only labels: that's a DAgger round.

**Benchmark** (`B`) runs every scenario (2 trials each, clear day) and seven 90 s free drives (clear day,
clear night, rain at dusk, fog, snow, and the highway by day and in rain at night) with the neural driver and scores them: passed scenarios, passed
without takeovers, meters of autonomous driving per takeover, contacts.

### The loop

    npm install                     # puppeteer-core, for the headless runs below
    npm run dev                     # in another terminal
    python3 -m venv .venv && .venv/bin/pip install -r train/requirements.txt

    npm run collect -- --seeds 101,102,103,104,105 --frames 6000        # expert data, 5 cities
    npm run train                                                        # -> models/policy.onnx
    npm run collect -- --seeds 201,202,203 --frames 4000 --neural --noise 0   # DAgger data
    npm run collect -- --seeds 301,302,303 --frames 5000 --highway 0.8        # mostly highway
    npm run collect -- --seeds 401,402,403 --frames 6000 --dense               # heavy traffic
    npm run train -- --init models/policy.pt --epochs 6                  # fine-tune on all of data/
    npm run bench -- --seeds 1,2,3 --out models/bench.json               # three cities, combined

`scripts/headless.mjs` runs the app in headless Chrome (set `CHROME=` if it isn't found), one browser per
city seed. A browser that loses its GPU context is restarted for the frames still owed, and
uploads retry. Training saves `models/policy.last.pt` every epoch: `npm run train -- --resume` continues
an interrupted run, and a data-loader failure after the machine sleeps restarts the epoch. Everything also works from the UI: **Collect**, then `npm run train`, then **Neural** and
**Benchmark**.

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
- **Highway (done).** A six-lane motorway loop around the city with MOBIL lane changing, three highway
  scenarios, and highway drives in data collection and the benchmark.
- Phase 3: realistic sensor model (noise, blur, exposure, latency).
- **Phase 4: neural driving policy (in progress).** Training labels, safety driver, sensor rig and
  label images, data collection, PyTorch training and ONNX export, the in-browser neural driver,
  DAgger and a benchmark. See [Learning to drive](#learning-to-drive).

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
| `src/path.js` | Polyline with arc length that vehicles follow (base of both route types) |
| `src/planner.js` | Routes and shared driving logic (curve speed, signals, yielding, IDM, crossing prediction, overtaking) + ego expert |
| `src/highway.js` | Highway loop geometry, routes along its lanes, MOBIL lane changing |
| `src/highwayMesh.js` | Highway rendering: road, markings, barrier, guardrails, lamps, gantries, trees |
| `src/sim.js` | Headless world: stepping, manual/expert control, contacts, weather conditions, scenario runner |
| `src/scenarios.js` | Scripted scenarios and their pass/fail checks |
| `src/input.js` | Keyboard / gamepad driving input |
| `src/debug.js` | Planner overlay |
| `src/weather.js` | Weather look and rain/snow particles |
| `src/recorder.js`, `src/zip.js` | Camera dataset recorder and a minimal ZIP writer |
| `src/labels.js` | Training labels from the expert: command, waypoints per command branch, target speed |
| `src/sensor.js` | Roof camera render target and semantic class / depth label images |
| `src/collect.js` | Streams training samples to the dev server; steering noise for recovery data |
| `src/safety.js` | Safety driver: supervises a learned driver, counts takeovers |
| `src/neural.js` | Neural driver: ONNX Runtime Web inference, waypoint following |
| `src/sessions.js` | Automated collection episodes and the benchmark |
| `src/neuralview.js` | Neural driver panel (attention, segmentation, steering chart, steering wheels) and scorecard |
| `train/` | PyTorch dataset, model, training and ONNX export |
| `scripts/serve.py`, `scripts/headless.mjs` | Dev server with the data upload API; headless Chrome runner |
| `src/main.js` | Renderer, sky/sun, post-processing, cameras, HUD, controls |

## Assets

Textures in `assets/tex/` are from [Poly Haven](https://polyhaven.com) (CC0): asphalt_02, concrete_pavement,
concrete_floor_worn_02, brick_wall_02, concrete_panels, clay_plaster, bark_brown_02.
The pedestrian model (Xbot) is loaded at runtime from the three.js examples via jsDelivr.
