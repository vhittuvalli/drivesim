// Ego-centric camera: renders a low-res, car-aligned view of the world into raw RGB pixels.
// The car sits near the bottom of the frame facing up, so most of the image is the road ahead.

import { GRASS } from './world.js';

export const SW = 64;
export const SH = 64;
export const SCALE = 2; // px per meter -> 32 m wide, 30 m ahead
export const CAR_Y = 60;

export class Camera {
  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = SW;
    this.canvas.height = SH;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.rgb = new Uint8Array(SW * SH * 3);
  }

  capture(world, car) {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = GRASS;
    ctx.fillRect(0, 0, SW, SH);
    ctx.translate(SW / 2, CAR_Y);
    ctx.rotate(-Math.PI / 2 - car.h);
    ctx.scale(SCALE, SCALE);
    ctx.translate(-car.x, -car.y);
    world.draw(ctx, { sensor: true });

    const rgba = ctx.getImageData(0, 0, SW, SH).data;
    const rgb = new Uint8Array(SW * SH * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    }
    this.rgb = rgb;
    return rgb;
  }
}

// Footprint of the camera in the car's frame (meters; forward = -y), for drawing on the main view.
export const FOOTPRINT = {
  x: -SW / 2 / SCALE,
  y: -CAR_Y / SCALE,
  w: SW / SCALE,
  h: SH / SCALE,
};
