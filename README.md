# DriveSim

End-to-end self-driving sandbox that runs entirely in the browser.

**Pipeline:** procedural world (road, lanes, traffic lights, cones) → 64×64 ego camera pixels → CNN (TF.js) → steering + throttle → kinematic bicycle model → next frame.

## Run

    npm run dev        # or: python3 -m http.server 8000
    open http://localhost:8000

## Workflow

1. **Collect**: the Expert (pure pursuit + ground-truth rules) drives with steering noise. Click *Start recording* (8× sim speed helps). About 3–5k samples is enough.
2. **Train**: behavioral cloning on (pixels, speed) → expert (steer, throttle).
3. **Drive**: switch to *Neural net*. Metrics track distance per incident.
4. **Improve (DAgger)**: keep recording while the NN drives. The expert still labels every frame, so the model learns to recover from its own mistakes. Retrain.

The *safety supervisor* is a rule-based layer that can only brake (red lights and obstacles). Turn it off to see what the NN alone has learned.

## Layout

| File | Role |
|---|---|
| `src/world.js` | Track generation, lights, cones, projection helpers, rendering |
| `src/sensor.js` | Ego camera → raw RGB pixels |
| `src/car.js` | Vehicle dynamics |
| `src/expert.js` | Teacher driver + safety supervisor |
| `src/brain.js` | Dataset ring buffer, CNN, training loop |
| `src/main.js` | Sim loop, rendering, UI |

## Deploy

Static site with no build step. Push to GitHub Pages, Vercel, or Netlify as-is.
