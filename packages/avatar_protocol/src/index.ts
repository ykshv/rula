export type BranchState =
  | "listening"
  | "speculative"
  | "committed"
  | "speaking"
  | "interrupted"
  | "discarded";

export type InboundEventKind = "client.ready" | "user.interrupt" | "avatar.state_ack";

export type OutboundEventKind =
  | "turn.state"
  | "turn.trace"
  | "turn.partial_transcript"
  | "turn.final_transcript"
  | "assistant.partial_text"
  | "assistant.speech_segment"
  | "avatar.state"
  | "avatar.blendshape_frame"
  | "avatar.emotion"
  | "avatar.gesture"
  | "error";

export interface StreamEnvelope<TKind extends string = string, TPayload = Record<string, unknown>> {
  session_id: string;
  turn_id: number;
  generation_id: number;
  branch_state: BranchState;
  seq: number;
  kind: TKind;
  pts_ms?: number | null;
  emitted_at_ms?: number;
  payload: TPayload;
}

export interface BlendshapeFramePayload {
  values: Record<string, number>;
  emotion?: string;
}

export interface AvatarStatePayload {
  state: "listening" | "thinking" | "speaking" | "interrupted" | "idle";
  shift_probability?: number;
  backchannel_probability?: number;
}

export interface GesturePayload {
  name: "nod" | "gaze_shift" | "posture_shift" | "micro_expression";
  intensity: number;
}

export type AvatarProtocolEvent =
  | StreamEnvelope<"avatar.blendshape_frame", BlendshapeFramePayload>
  | StreamEnvelope<"avatar.state", AvatarStatePayload>
  | StreamEnvelope<"avatar.gesture", GesturePayload>
  | StreamEnvelope<OutboundEventKind, Record<string, unknown>>;

export function isCurrentGeneration(
  event: Pick<StreamEnvelope, "session_id" | "generation_id">,
  current: Pick<StreamEnvelope, "session_id" | "generation_id">,
): boolean {
  return event.session_id === current.session_id && event.generation_id === current.generation_id;
}

export function hasPts(event: StreamEnvelope): event is StreamEnvelope & { pts_ms: number } {
  return typeof event.pts_ms === "number" && Number.isFinite(event.pts_ms);
}
