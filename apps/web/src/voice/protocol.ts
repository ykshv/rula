import type { StreamEnvelope } from "@ru-local-avatar/avatar-protocol";

export interface FaceFramePayload {
  pts_ms: number;
  values: Record<string, number>;
}

export interface BlendshapeBatchPayload {
  frames: FaceFramePayload[];
}

export type AvatarState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "interrupted";

export function parseEnvelope(raw: Uint8Array): StreamEnvelope | null {
  try {
    const decoded = JSON.parse(new TextDecoder().decode(raw)) as StreamEnvelope;
    if (
      typeof decoded !== "object" ||
      decoded === null ||
      typeof decoded.kind !== "string" ||
      typeof decoded.generation_id !== "number"
    ) {
      return null;
    }
    return decoded;
  } catch {
    return null;
  }
}
