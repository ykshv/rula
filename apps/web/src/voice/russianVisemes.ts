import type { FaceFramePayload } from "./protocol";

type Viseme = "closed" | "a" | "i" | "u" | "e" | "o" | "soft" | "neutral";

const FRAME_STEP_MS = 33;
const MIN_HOLD_MS = 66;

const VOWELS: Record<string, Viseme> = {
  а: "a",
  я: "a",
  и: "i",
  ы: "i",
  у: "u",
  ю: "u",
  э: "e",
  е: "e",
  о: "o",
  ё: "o",
};

const CONSONANTS: Record<string, Viseme> = {
  б: "closed",
  м: "closed",
  п: "closed",
  в: "soft",
  ф: "soft",
  ж: "u",
  ш: "u",
  щ: "u",
  ч: "u",
};

const PUNCTUATION_PAUSE_MS: Record<string, number> = {
  ",": 90,
  ".": 150,
  "!": 150,
  "?": 150,
  ";": 120,
  ":": 120,
};

export function buildRussianVisemeFrames(text: string, basePtsMs: number): FaceFramePayload[] {
  const frames: FaceFramePayload[] = [
    { pts_ms: Math.max(0, basePtsMs), values: valuesFor("neutral") },
  ];
  let cursor = Math.max(0, basePtsMs);
  let lastViseme: Viseme = "neutral";

  for (const raw of text.toLowerCase()) {
    if (raw in PUNCTUATION_PAUSE_MS) {
      cursor += PUNCTUATION_PAUSE_MS[raw];
      frames.push({ pts_ms: cursor, values: valuesFor("neutral") });
      lastViseme = "neutral";
      continue;
    }
    if (/\s/u.test(raw)) {
      cursor += 35;
      continue;
    }

    const viseme = VOWELS[raw] ?? CONSONANTS[raw] ?? "neutral";
    const duration = durationFor(raw, viseme, lastViseme);
    cursor += duration;

    if (viseme !== "neutral") {
      frames.push({ pts_ms: cursor - Math.min(35, duration), values: valuesFor(viseme, 0.72) });
    }
    frames.push({ pts_ms: cursor, values: valuesFor(viseme) });
    lastViseme = viseme;
  }

  frames.push({ pts_ms: cursor + 80, values: valuesFor("neutral") });
  return normalizeFrameCadence(frames);
}

function durationFor(char: string, viseme: Viseme, previous: Viseme): number {
  if (VOWELS[char]) {
    return viseme === previous ? 80 : 95;
  }
  if (viseme === "closed") {
    return 55;
  }
  if (viseme === "soft" || viseme === "u") {
    return 62;
  }
  return 42;
}

function normalizeFrameCadence(frames: FaceFramePayload[]): FaceFramePayload[] {
  if (frames.length <= 1) {
    return frames;
  }
  const out: FaceFramePayload[] = [frames[0]];
  for (let i = 1; i < frames.length; i += 1) {
    const prev = out[out.length - 1];
    const next = frames[i];
    if (next.pts_ms - prev.pts_ms >= MIN_HOLD_MS) {
      const bridgePts = Math.max(prev.pts_ms + FRAME_STEP_MS, next.pts_ms - FRAME_STEP_MS);
      if (bridgePts > prev.pts_ms && bridgePts < next.pts_ms) {
        out.push({ pts_ms: bridgePts, values: mixValues(prev.values, next.values, 0.5) });
      }
    }
    out.push(next);
  }
  return out;
}

function valuesFor(viseme: Viseme, intensity = 1): Record<string, number> {
  const scale = (value: number) => Math.min(1, Math.max(0, value * intensity));
  switch (viseme) {
    case "closed":
      return {
        mouthClose: scale(0.92),
        jawOpen: scale(0.02),
      };
    case "a":
      return {
        jawOpen: scale(0.78),
        mouthLowerDownLeft: scale(0.42),
        mouthLowerDownRight: scale(0.42),
        mouthUpperUpLeft: scale(0.16),
        mouthUpperUpRight: scale(0.16),
      };
    case "i":
      return {
        jawOpen: scale(0.24),
        mouthStretchLeft: scale(0.72),
        mouthStretchRight: scale(0.72),
        mouthSmileLeft: scale(0.16),
        mouthSmileRight: scale(0.16),
      };
    case "u":
      return {
        jawOpen: scale(0.18),
        mouthFunnel: scale(0.76),
        mouthPucker: scale(0.7),
      };
    case "e":
      return {
        jawOpen: scale(0.38),
        mouthStretchLeft: scale(0.48),
        mouthStretchRight: scale(0.48),
      };
    case "o":
      return {
        jawOpen: scale(0.48),
        mouthFunnel: scale(0.72),
        mouthPucker: scale(0.36),
      };
    case "soft":
      return {
        mouthClose: scale(0.34),
        jawOpen: scale(0.12),
        mouthLowerDownLeft: scale(0.24),
        mouthLowerDownRight: scale(0.24),
      };
    case "neutral":
    default:
      return {
        jawOpen: 0,
        mouthClose: 0,
        mouthFunnel: 0,
        mouthPucker: 0,
      };
  }
}

function mixValues(
  a: Record<string, number>,
  b: Record<string, number>,
  alpha: number,
): Record<string, number> {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const mixed: Record<string, number> = {};
  for (const key of keys) {
    const av = a[key] ?? 0;
    const bv = b[key] ?? 0;
    mixed[key] = av + (bv - av) * alpha;
  }
  return mixed;
}
