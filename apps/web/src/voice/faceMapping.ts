/**
 * ARKit blendshape weights (Audio2Face-3D output) -> VRM 0.x expression
 * weights. The default avatar exposes the classic VRM preset morphs
 * (aa/ih/ou/ee/oh, blink, emotion presets), so the 52 ARKit channels are
 * folded into visemes + eye/emotion accents.
 */

export interface VrmExpressionWeights {
  aa: number;
  ih: number;
  ou: number;
  ee: number;
  oh: number;
  blinkLeft: number;
  blinkRight: number;
  happy: number;
  lookUp: number;
  lookDown: number;
  lookLeft: number;
  lookRight: number;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

export function arkitToVrm(values: Record<string, number>): VrmExpressionWeights {
  const v = (name: string) => values[name] ?? 0;

  const jaw = v("jawOpen");
  const round = Math.max(v("mouthFunnel"), v("mouthPucker"));
  const smile = (v("mouthSmileLeft") + v("mouthSmileRight")) / 2;
  const stretch = (v("mouthStretchLeft") + v("mouthStretchRight")) / 2;
  const lowerDown = (v("mouthLowerDownLeft") + v("mouthLowerDownRight")) / 2;
  const upperUp = (v("mouthUpperUpLeft") + v("mouthUpperUpRight")) / 2;
  const close = v("mouthClose");

  const openness = clamp01(jaw * 1.35 + (lowerDown + upperUp) * 0.25 - close * 0.8);
  const spread = clamp01(smile * 0.7 + stretch * 0.5);

  const aa = clamp01(openness * (1 - round * 0.75));
  const oh = clamp01(Math.min(openness, round) * 1.35);
  const ou = clamp01(round * (1 - openness) * 1.25);
  const ee = clamp01(spread * (1 - openness * 0.6) * 1.1);
  const ih = clamp01(openness * 0.45 * (0.4 + spread));

  const lookUp = clamp01((v("eyeLookUpLeft") + v("eyeLookUpRight")) / 2);
  const lookDown = clamp01((v("eyeLookDownLeft") + v("eyeLookDownRight")) / 2);
  const lookLeft = clamp01((v("eyeLookOutLeft") + v("eyeLookInRight")) / 2);
  const lookRight = clamp01((v("eyeLookOutRight") + v("eyeLookInLeft")) / 2);

  return {
    aa,
    ih,
    ou,
    ee,
    oh,
    blinkLeft: clamp01(v("eyeBlinkLeft")),
    blinkRight: clamp01(v("eyeBlinkRight")),
    happy: clamp01(smile * 0.35),
    lookUp,
    lookDown,
    lookLeft,
    lookRight,
  };
}

export const NEUTRAL_EXPRESSION: VrmExpressionWeights = {
  aa: 0,
  ih: 0,
  ou: 0,
  ee: 0,
  oh: 0,
  blinkLeft: 0,
  blinkRight: 0,
  happy: 0,
  lookUp: 0,
  lookDown: 0,
  lookLeft: 0,
  lookRight: 0,
};
