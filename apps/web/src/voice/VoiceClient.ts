import {
  ConnectionState,
  RemoteAudioTrack,
  Room,
  RoomEvent,
  Track,
} from "livekit-client";
import type { StreamEnvelope } from "@ru-local-avatar/avatar-protocol";
import { FaceScheduler } from "./FaceScheduler";
import { MobileEchoSafeMicGate } from "./MobileEchoSafeMicGate";
import { parseEnvelope, type AvatarState, type BlendshapeBatchPayload } from "./protocol";
import { buildRussianVisemeFrames } from "./russianVisemes";

export interface VoiceSessionInfo {
  livekit_ws_url: string;
  livekit_token: string;
  room_name: string;
  session_id: string;
}

export interface VoiceCallbacks {
  onState: (state: AvatarState) => void;
  onPartialTranscript: (text: string) => void;
  onFinalTranscript: (text: string) => void;
  onAssistantSegment: (text: string) => void;
  onEvent?: (event: StreamEnvelope) => void;
  onError: (message: string) => void;
  onConnectionChange: (connected: boolean) => void;
}

const DATA_TOPIC = "avatar-protocol";
const ONSET_THRESHOLD = 0.015;
const ONSET_FALLBACK_MS = 180;
const MICROPHONE_OPTIONS = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

export class VoiceClient {
  readonly faceScheduler = new FaceScheduler();
  private room: Room | null = null;
  private audioElement: HTMLAudioElement | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private analyserBuffer: Float32Array<ArrayBuffer> | null = null;
  private monitorHandle = 0;
  private onsetFallbackHandle = 0;
  private currentSessionId = "";
  private currentGeneration = -1;
  private awaitingOnset = false;
  private readonly echoSafeMicGate: MobileEchoSafeMicGate;

  constructor(private callbacks: VoiceCallbacks) {
    this.echoSafeMicGate = new MobileEchoSafeMicGate(
      (enabled) => this.setMicrophoneEnabled(enabled),
      (message) => this.callbacks.onError(message),
    );
  }

  get connected(): boolean {
    return this.room?.state === ConnectionState.Connected;
  }

  async connect(info: VoiceSessionInfo): Promise<void> {
    if (this.room) {
      await this.disconnect();
    }
    this.resetSessionState(info.session_id);
    const room = new Room();
    this.room = room;

    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === Track.Kind.Audio && track instanceof RemoteAudioTrack) {
        this.attachRemoteAudio(track);
      }
    });
    room.on(RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
      if (topic === DATA_TOPIC) {
        const envelope = parseEnvelope(payload);
        if (envelope) {
          this.handleEnvelope(envelope);
        }
      }
    });
    room.on(RoomEvent.Disconnected, () => {
      this.callbacks.onConnectionChange(false);
    });

    await room.connect(info.livekit_ws_url, info.livekit_token, { autoSubscribe: true });
    await this.setMicrophoneEnabled(true);
    this.callbacks.onConnectionChange(true);
  }

  async disconnect(): Promise<void> {
    this.echoSafeMicGate.reset();
    this.resetSessionState("");
    window.cancelAnimationFrame(this.monitorHandle);
    this.monitorHandle = 0;
    this.clearOnsetFallback();
    if (this.audioContext) {
      await this.audioContext.close().catch(() => undefined);
      this.audioContext = null;
    }
    this.audioElement?.remove();
    this.audioElement = null;
    if (this.room) {
      await this.room.disconnect();
      this.room = null;
    }
    this.callbacks.onConnectionChange(false);
  }

  private attachRemoteAudio(track: RemoteAudioTrack) {
    const element = track.attach();
    element.autoplay = true;
    element.style.display = "none";
    document.body.appendChild(element);
    this.audioElement = element;

    const stream = new MediaStream([track.mediaStreamTrack]);
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    this.audioContext = context;
    this.analyser = analyser;
    this.analyserBuffer = new Float32Array(analyser.fftSize);
    this.startOnsetMonitor();
  }

  private startOnsetMonitor() {
    const tick = () => {
      this.monitorHandle = window.requestAnimationFrame(tick);
      if (!this.analyser || !this.analyserBuffer || !this.awaitingOnset) {
        return;
      }
      this.analyser.getFloatTimeDomainData(this.analyserBuffer);
      let sum = 0;
      for (let i = 0; i < this.analyserBuffer.length; i += 1) {
        sum += this.analyserBuffer[i] * this.analyserBuffer[i];
      }
      const rms = Math.sqrt(sum / this.analyserBuffer.length);
      if (rms > ONSET_THRESHOLD) {
        this.faceScheduler.anchor(performance.now());
        this.clearOnsetFallback();
        this.awaitingOnset = false;
      }
    };
    tick();
  }

  private handleEnvelope(envelope: StreamEnvelope) {
    if (this.currentSessionId && envelope.session_id !== this.currentSessionId) {
      return; // old room event delivered after reconnect
    }
    if (!this.currentSessionId) {
      this.currentSessionId = envelope.session_id;
    }
    if (envelope.generation_id < this.currentGeneration) {
      return; // stale
    }
    if (envelope.generation_id > this.currentGeneration) {
      this.currentGeneration = envelope.generation_id;
    }
    this.callbacks.onEvent?.(envelope);

    switch (envelope.kind) {
      case "avatar.state": {
        const state = (envelope.payload as { state?: string }).state ?? "idle";
        this.echoSafeMicGate.onAvatarState(state as AvatarState);
        if (state === "interrupted") {
          this.faceScheduler.interrupt();
          this.awaitingOnset = false;
          this.clearOnsetFallback();
        }
        if (state === "speaking") {
          this.faceScheduler.beginGeneration(envelope.generation_id);
          this.awaitingOnset = true;
          this.armOnsetFallback();
          void this.audioContext?.resume();
        }
        this.callbacks.onState(state as AvatarState);
        break;
      }
      case "avatar.blendshape_frame": {
        const payload = envelope.payload as unknown as BlendshapeBatchPayload;
        if (Array.isArray(payload.frames)) {
          this.faceScheduler.push(envelope.generation_id, payload.frames);
        }
        break;
      }
      case "turn.partial_transcript":
        this.callbacks.onPartialTranscript(String((envelope.payload as { text?: string }).text ?? ""));
        break;
      case "turn.final_transcript":
        this.callbacks.onFinalTranscript(String((envelope.payload as { text?: string }).text ?? ""));
        break;
      case "assistant.speech_segment":
        {
          const text = String((envelope.payload as { text?: string }).text ?? "");
          this.faceScheduler.pushFallback(
            envelope.generation_id,
            buildRussianVisemeFrames(text, envelope.pts_ms ?? 0),
          );
          this.callbacks.onAssistantSegment(text);
        }
        break;
      case "error":
        this.callbacks.onError(String((envelope.payload as { message?: string }).message ?? "pipeline error"));
        break;
      default:
        break;
    }
  }

  private async setMicrophoneEnabled(enabled: boolean): Promise<void> {
    if (!this.room) {
      return;
    }
    await this.room.localParticipant.setMicrophoneEnabled(enabled, MICROPHONE_OPTIONS);
  }

  private armOnsetFallback() {
    this.clearOnsetFallback();
    this.onsetFallbackHandle = window.setTimeout(() => {
      this.onsetFallbackHandle = 0;
      if (this.awaitingOnset && !this.faceScheduler.anchored) {
        this.faceScheduler.anchor(performance.now());
        this.awaitingOnset = false;
      }
    }, ONSET_FALLBACK_MS);
  }

  private clearOnsetFallback() {
    if (this.onsetFallbackHandle !== 0) {
      window.clearTimeout(this.onsetFallbackHandle);
      this.onsetFallbackHandle = 0;
    }
  }

  private resetSessionState(sessionId: string) {
    this.currentSessionId = sessionId;
    this.currentGeneration = -1;
    this.awaitingOnset = false;
    this.clearOnsetFallback();
    this.faceScheduler.interrupt();
  }
}
