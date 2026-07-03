import type { FaceFramePayload } from "./protocol";

/**
 * Schedules Audio2Face frames against the actually-heard audio.
 *
 * pts_ms is relative to the first audio sample of the current generation.
 * The anchor (t0) is set by an energy-onset detector running on the remote
 * audio track, so audio-face drift stays bounded by the onset detection
 * error (one analyser hop, ~11 ms) rather than network jitter guesses.
 */
export class FaceScheduler {
  private neuralFrames: FaceFramePayload[] = [];
  private fallbackFrames: FaceFramePayload[] = [];
  private generationId = -1;
  private anchorMs: number | null = null;
  private lastValues: Record<string, number> = {};
  private decay = 0;

  beginGeneration(generationId: number) {
    if (generationId !== this.generationId) {
      this.generationId = generationId;
      this.neuralFrames = [];
      this.fallbackFrames = [];
      this.anchorMs = null;
      this.decay = 0;
    }
  }

  push(generationId: number, frames: FaceFramePayload[]) {
    if (generationId < this.generationId) {
      return; // stale generation, dropped silently
    }
    this.beginGeneration(generationId);
    this.neuralFrames.push(...frames);
    this.neuralFrames.sort(byPts);
  }

  pushFallback(generationId: number, frames: FaceFramePayload[]) {
    if (generationId < this.generationId) {
      return;
    }
    this.beginGeneration(generationId);
    this.fallbackFrames.push(...frames);
    this.fallbackFrames.sort(byPts);
  }

  /** Called by the audio-onset detector when the response actually starts sounding. */
  anchor(nowMs: number) {
    if (this.anchorMs === null) {
      this.anchorMs = nowMs;
    }
  }

  get anchored(): boolean {
    return this.anchorMs !== null;
  }

  interrupt() {
    this.neuralFrames = [];
    this.fallbackFrames = [];
    this.anchorMs = null;
    this.decay = 1;
  }

  /** Sample the mouth pose for the current time; lerps between 30 fps frames. */
  sample(nowMs: number): Record<string, number> | null {
    if (this.decay > 0) {
      this.decay = Math.max(0, this.decay - 0.12);
      const faded: Record<string, number> = {};
      for (const [key, value] of Object.entries(this.lastValues)) {
        faded[key] = value * this.decay;
      }
      this.lastValues = faded;
      return faded;
    }
    if (this.anchorMs === null || (this.neuralFrames.length === 0 && this.fallbackFrames.length === 0)) {
      return null;
    }
    const pts = nowMs - this.anchorMs;
    const neural = sampleFrames(this.neuralFrames, pts, this.lastValues);
    const fallback = sampleFrames(this.fallbackFrames, pts, this.lastValues);
    let values = neural ?? fallback;

    if (fallback && (!neural || mouthActivity(neural) < 0.035)) {
      values = fallback;
    }
    if (!values) {
      return null;
    }
    this.lastValues = values;
    return values;
  }
}

function sampleFrames(
  frames: FaceFramePayload[],
  pts: number,
  lastValues: Record<string, number>,
): Record<string, number> | null {
  if (frames.length === 0) {
    return null;
  }
  while (frames.length > 1 && frames[1].pts_ms <= pts) {
    frames.shift();
  }
  const current = frames[0];
  if (current.pts_ms > pts + 400) {
    return Object.keys(lastValues).length > 0 ? lastValues : null;
  }
  let values = current.values;
  const next = frames[1];
  if (next && next.pts_ms > current.pts_ms) {
    const alpha = Math.min(1, Math.max(0, (pts - current.pts_ms) / (next.pts_ms - current.pts_ms)));
    const mixed: Record<string, number> = {};
    for (const key of new Set([...Object.keys(current.values), ...Object.keys(next.values)])) {
      const a = current.values[key] ?? 0;
      const b = next.values[key] ?? a;
      mixed[key] = a + (b - a) * alpha;
    }
    values = mixed;
  }
  return values;
}

function mouthActivity(values: Record<string, number>): number {
  return Math.max(
    values.jawOpen ?? 0,
    values.mouthFunnel ?? 0,
    values.mouthPucker ?? 0,
    values.mouthClose ?? 0,
    values.mouthStretchLeft ?? 0,
    values.mouthStretchRight ?? 0,
    values.mouthSmileLeft ?? 0,
    values.mouthSmileRight ?? 0,
  );
}

function byPts(a: FaceFramePayload, b: FaceFramePayload): number {
  return a.pts_ms - b.pts_ms;
}
