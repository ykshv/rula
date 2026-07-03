import React from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronUp,
  Mic,
  MicOff,
  RefreshCw,
} from "lucide-react";
import type { StreamEnvelope } from "@ru-local-avatar/avatar-protocol";
import { VrmAvatarStage, type AvatarStatus } from "./VrmAvatarStage";
import { VoiceClient } from "./voice/VoiceClient";
import "./styles.css";

const AGENT_BASE_URL = import.meta.env.VITE_AGENT_BASE_URL ?? "";

interface RuntimeStatus {
  mode: string;
  profile_path: string;
  voice_preset: string;
  avatar_url: string;
  artifacts: Record<string, { ok: boolean; path?: string; size_gb?: number; detail?: string }>;
  services: Record<
    string,
    {
      ok: boolean;
      detail?: string;
      base_url?: string;
      host?: string;
      port?: number;
      free_gb?: number;
      min_free_gb?: number;
    }
  >;
  ready: {
    text_chat: boolean;
    voice_avatar: boolean;
    all_artifacts: boolean;
  };
}

interface SessionResponse {
  session_id: string;
  room_name: string;
  livekit_token: string;
  livekit_ws_url: string;
  livekit_public_ws_url: string;
  avatar_url: string;
  voice_enabled: boolean;
  voice_disabled_reason: string;
  barge_in_enabled: boolean;
}

interface VoiceLine {
  role: "user" | "assistant";
  text: string;
  final: boolean;
}

interface EventLogEntry {
  id: string;
  event: StreamEnvelope;
  firstEmittedAt: number;
  firstReceivedAt: number;
  emittedAt: number;
  receivedAt: number;
  count: number;
  firstSeq: number;
  lastSeq: number;
}

interface TurnTracePayload {
  latency_tier?: string;
  playback_policy?: string;
  from_speech_end_ms?: Record<string, number | null | undefined>;
  phase_ms?: Record<string, number | null | undefined>;
  tts?: {
    unit_count?: number;
    cache_hits?: number;
    cache_misses?: number;
  };
  reliability?: {
    underruns?: number;
    stale_drops?: number;
  };
}

interface RuntimeCheck {
  label: string;
  ok?: boolean;
  detail?: string;
}

const EVENT_LOG_LIMIT = 48;
const EVENT_LOG_FLUSH_MS = 120;
const AGGREGATED_EVENT_KINDS = new Set([
  "avatar.blendshape_frame",
  "turn.partial_transcript",
]);

function App() {
  const [runtime, setRuntime] = React.useState<RuntimeStatus | null>(null);
  const [session, setSession] = React.useState<SessionResponse | null>(null);
  const [events, setEvents] = React.useState<EventLogEntry[]>([]);
  const [voiceLines, setVoiceLines] = React.useState<VoiceLine[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [avatarStatus, setAvatarStatus] = React.useState<AvatarStatus>("offline");
  const [voiceConnected, setVoiceConnected] = React.useState(false);
  const [voiceBusy, setVoiceBusy] = React.useState(false);
  const [bargeInEnabled, setBargeInEnabled] = React.useState(false);
  const voiceClientRef = React.useRef<VoiceClient | null>(null);
  const voiceStartInFlightRef = React.useRef(false);
  const eventLogRef = React.useRef<EventLogEntry[]>([]);
  const eventFlushHandleRef = React.useRef<number>(0);

  const appendVoiceLine = React.useCallback((line: VoiceLine) => {
    setVoiceLines((current) => {
      const next = [...current];
      const last = next[next.length - 1];
      if (last && last.role === line.role && !last.final) {
        next[next.length - 1] = line;
      } else {
        next.push(line);
      }
      return next.slice(-12);
    });
  }, []);

  const flushEventLog = React.useCallback(() => {
    if (eventFlushHandleRef.current !== 0) {
      window.clearTimeout(eventFlushHandleRef.current);
      eventFlushHandleRef.current = 0;
    }
    setEvents([...eventLogRef.current]);
  }, []);

  const resetEventLog = React.useCallback(() => {
    if (eventFlushHandleRef.current !== 0) {
      window.clearTimeout(eventFlushHandleRef.current);
      eventFlushHandleRef.current = 0;
    }
    eventLogRef.current = [];
    setEvents([]);
  }, []);

  const appendEvent = React.useCallback(
    (event: StreamEnvelope) => {
      eventLogRef.current = reduceEventLog(eventLogRef.current, event, Date.now());
      if (!isAggregatedEvent(event)) {
        flushEventLog();
        return;
      }
      if (eventFlushHandleRef.current === 0) {
        eventFlushHandleRef.current = window.setTimeout(flushEventLog, EVENT_LOG_FLUSH_MS);
      }
    },
    [flushEventLog],
  );

  React.useEffect(
    () => () => {
      if (eventFlushHandleRef.current !== 0) {
        window.clearTimeout(eventFlushHandleRef.current);
      }
    },
    [],
  );

  const ensureVoiceClient = React.useCallback(() => {
    if (voiceClientRef.current) {
      return voiceClientRef.current;
    }
    const client = new VoiceClient({
      onState: (state) => setAvatarStatus(state as AvatarStatus),
      onPartialTranscript: (text) => appendVoiceLine({ role: "user", text, final: false }),
      onFinalTranscript: (text) => appendVoiceLine({ role: "user", text, final: true }),
      onAssistantSegment: (text) =>
        setVoiceLines((current) => {
          const next = [...current];
          const last = next[next.length - 1];
          if (last && last.role === "assistant" && !last.final) {
            next[next.length - 1] = { ...last, text: `${last.text} ${text}` };
          } else {
            next.push({ role: "assistant", text, final: false });
          }
          return next.slice(-12);
        }),
      onEvent: appendEvent,
      onError: (message) => setError(message),
      onConnectionChange: (connected) => {
        setVoiceConnected(connected);
        if (!connected) {
          setAvatarStatus("ready");
        }
      },
    });
    voiceClientRef.current = client;
    return client;
  }, [appendEvent, appendVoiceLine]);

  const refreshRuntime = React.useCallback(async () => {
    const next = await apiGet<RuntimeStatus>("/api/runtime/status");
    setRuntime(next);
    setAvatarStatus((current) =>
      current === "offline" || current === "ready"
        ? next.ready.text_chat
          ? "ready"
          : "offline"
        : current,
    );
    return next;
  }, []);

  React.useEffect(() => {
    refreshRuntime().catch((refreshError: unknown) => {
      setError(toErrorMessage(refreshError));
      setAvatarStatus("offline");
    });
  }, [refreshRuntime]);

  const startVoice = React.useCallback(async () => {
    if (voiceStartInFlightRef.current) {
      return;
    }
    voiceStartInFlightRef.current = true;
    setError(null);
    setVoiceBusy(true);
    try {
      const client = ensureVoiceClient();
      if (client.connected) {
        setAvatarStatus("idle");
        return;
      }
      resetEventLog();
      setVoiceLines([]);
      // Voice always needs a fresh session so the agent worker joins the room.
      const created = await apiPost<SessionResponse>("/api/sessions", {
        barge_in_enabled: bargeInEnabled,
      });
      setSession(created);
      if (!created.voice_enabled || !created.livekit_token) {
        setError(`Voice pipeline disabled: ${created.voice_disabled_reason || "not ready"}`);
        return;
      }
      await client.connect({
        livekit_ws_url: created.livekit_public_ws_url || created.livekit_ws_url,
        livekit_token: created.livekit_token,
        room_name: created.room_name,
        session_id: created.session_id,
      });
      setAvatarStatus("idle");
    } catch (voiceError: unknown) {
      setError(toErrorMessage(voiceError));
    } finally {
      voiceStartInFlightRef.current = false;
      setVoiceBusy(false);
    }
  }, [bargeInEnabled, ensureVoiceClient, resetEventLog]);

  const stopVoice = React.useCallback(async () => {
    setVoiceBusy(true);
    try {
      await voiceClientRef.current?.disconnect();
    } finally {
      setVoiceBusy(false);
      setAvatarStatus("ready");
    }
  }, []);

  const voiceReady = Boolean(runtime?.ready.voice_avatar);
  const latestUserLine = latestVoiceLine(voiceLines, "user");
  const latestAssistantLine = latestVoiceLine(voiceLines, "assistant");

  return (
    <main className="app-shell">
      <section className="stage-band" aria-label="Avatar session">
        <div className="workspace">
          <VrmAvatarStage
            faceScheduler={voiceConnected ? (voiceClientRef.current?.faceScheduler ?? null) : null}
            status={avatarStatus}
            avatarUrl={runtime?.avatar_url}
            subtitle={latestAssistantLine?.text}
          />

          <aside className="control-panel" aria-label="Session telemetry and controls">
            <section className="voice-panel" aria-label="Voice session">
              <div className="panel-heading">
                <div>
                  <h2>Voice</h2>
                  <span>{session ? `session ${shortId(session.session_id)}` : "no session"}</span>
                </div>
                <strong>{voiceConnected ? "live" : voiceReady ? "ready" : "disabled"}</strong>
              </div>
              {latestUserLine && <LatestUserTranscript line={latestUserLine} />}
              <button
                type="button"
                className="session-switch"
                role="switch"
                aria-checked={bargeInEnabled}
                disabled={voiceConnected || voiceBusy}
                onClick={() => setBargeInEnabled((current) => !current)}
              >
                <span className="session-switch-label">Перебивание</span>
                <span className="switch-track" aria-hidden="true">
                  <span className="switch-thumb" />
                </span>
              </button>
              <div className="controls">
                {!voiceConnected ? (
                  <button
                    className="primary-action"
                    type="button"
                    onClick={() => void startVoice()}
                    disabled={!voiceReady || voiceBusy}
                  >
                    <Mic size={18} />
                    Start voice
                  </button>
                ) : (
                  <button
                    className="secondary-action"
                    type="button"
                    onClick={() => void stopVoice()}
                    disabled={voiceBusy}
                  >
                    <MicOff size={18} />
                    Stop voice
                  </button>
                )}
              </div>
              {!voiceReady && (
                <p className="voice-disabled-note">
                  Voice unavailable: {voiceUnavailableReason(runtime)}
                </p>
              )}
              {error && (
                <div className="error-box" role="alert">
                  <AlertTriangle size={18} />
                  <span>{error}</span>
                </div>
              )}
            </section>

            <EventLog entries={events} />
            <RuntimePanel runtime={runtime} onRefresh={refreshRuntime} />
          </aside>
        </div>
      </section>
    </main>
  );
}

function LatestUserTranscript({ line }: { line: VoiceLine }) {
  return (
    <div className="latest-user-transcript" aria-live="polite">
      <span>Вы</span>
      <p>{line.text}</p>
    </div>
  );
}

function RuntimePanel({
  runtime,
  onRefresh,
}: {
  runtime: RuntimeStatus | null;
  onRefresh: () => Promise<RuntimeStatus>;
}) {
  const [expanded, setExpanded] = React.useState(false);
  const checks = React.useMemo(() => runtimeChecks(runtime), [runtime]);
  const readyCount = checks.filter((check) => check.ok).length;
  const detailsId = React.useId();
  const summary = runtime ? `${readyCount} из ${checks.length} готовы` : "нет данных";

  return (
    <section
      className="runtime-panel"
      data-expanded={expanded ? "true" : "false"}
      aria-label="Состояние сервисов и моделей"
    >
      <div className="panel-heading runtime-heading">
        <div className="runtime-title">
          <h2>Сервисы и модели</h2>
          <span>{summary}</span>
        </div>
        <div className="runtime-actions">
          <button
            type="button"
            className="icon-action"
            onClick={() => {
              void onRefresh();
            }}
            aria-label="Обновить состояние сервисов и моделей"
          >
            <RefreshCw size={17} />
          </button>
          <button
            type="button"
            className="icon-action disclosure-action"
            onClick={() => setExpanded((current) => !current)}
            aria-controls={detailsId}
            aria-expanded={expanded}
            aria-label={expanded ? "Скрыть проверки сервисов и моделей" : "Показать проверки сервисов и моделей"}
          >
            <ChevronUp size={17} />
          </button>
        </div>
      </div>

      <div id={detailsId} className="runtime-details" hidden={!expanded}>
        <div className="check-list">
          {checks.map((check) => (
            <CheckRow key={check.label} label={check.label} ok={check.ok} detail={check.detail} />
          ))}
        </div>
      </div>
    </section>
  );
}

function CheckRow({ label, ok, detail }: RuntimeCheck) {
  return (
    <div className="check-row" data-ok={ok ? "true" : "false"}>
      {ok ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
      <span>{label}</span>
      <code>{detail ?? "unknown"}</code>
    </div>
  );
}

function EventLog({ entries }: { entries: EventLogEntry[] }) {
  const totalEvents = entries.reduce((sum, entry) => sum + entry.count, 0);
  const latestAt = entries[0]?.emittedAt;
  const traceEntries = entries.filter((entry) => entry.event.kind === "turn.trace").slice(0, 4);
  const rawEntries = entries.filter((entry) => entry.event.kind !== "turn.trace");

  return (
    <section className="event-log" aria-label="Agent events">
      <div className="panel-heading">
        <div className="event-log-title">
          <h2>Agent events</h2>
          <span>{latestAt ? `updated ${formatClock(latestAt)}` : "waiting"}</span>
        </div>
        <span>{totalEvents > 0 ? `${totalEvents} events / ${entries.length} rows` : "waiting"}</span>
      </div>
      <div className="events">
        {entries.length === 0 ? (
          <p className="empty">No real agent events yet</p>
        ) : (
          <>
            {traceEntries.length > 0 && (
              <div className="turn-traces" aria-label="Turn latency timeline">
                {traceEntries.map((entry) => (
                  <TurnTraceRow key={entry.id} entry={entry} />
                ))}
              </div>
            )}
            {rawEntries.map((entry) => (
              <article
                key={entry.id}
                className="event-row"
              >
                <div>
                  <span>{entry.event.kind}</span>
                  <small>{eventSummary(entry)}</small>
                </div>
                <div className="event-meta">
                  <time dateTime={new Date(entry.emittedAt).toISOString()}>
                    {formatClock(entry.emittedAt)}
                  </time>
                  <code>{eventSequenceLabel(entry)}</code>
                </div>
              </article>
            ))}
          </>
        )}
      </div>
    </section>
  );
}

function TurnTraceRow({ entry }: { entry: EventLogEntry }) {
  const payload = entry.event.payload as TurnTracePayload;
  const firstAudio = payload.from_speech_end_ms?.first_audio;
  const eot = payload.from_speech_end_ms?.eot_commit;
  const ttsFirst = payload.from_speech_end_ms?.tts_first_chunk;
  const cacheHits = payload.tts?.cache_hits ?? 0;
  const cacheMisses = payload.tts?.cache_misses ?? 0;
  const underruns = payload.reliability?.underruns ?? 0;
  const staleDrops = payload.reliability?.stale_drops ?? 0;

  return (
    <article className="turn-trace-row">
      <div className="turn-trace-main">
        <span>turn {entry.event.turn_id} / gen {entry.event.generation_id}</span>
        <strong>{msValue(firstAudio)} first audio</strong>
      </div>
      <div className="turn-trace-grid">
        <code>{payload.latency_tier ?? "unknown"}</code>
        <span>EOT {msValue(eot)}</span>
        <span>TTS {msValue(ttsFirst)}</span>
        <span>cache {cacheHits}/{cacheHits + cacheMisses}</span>
        <span>underrun {underruns}</span>
        <span>stale {staleDrops}</span>
      </div>
    </article>
  );
}

function reduceEventLog(
  entries: EventLogEntry[],
  event: StreamEnvelope,
  receivedAt: number,
): EventLogEntry[] {
  const emittedAt = event.emitted_at_ms ?? receivedAt;
  const current = entries[0];
  if (current && canMergeEventLogEntry(current, event)) {
    return [
      {
        ...current,
        event,
        emittedAt,
        receivedAt,
        count: current.count + 1,
        lastSeq: event.seq,
      },
      ...entries.slice(1),
    ];
  }

  const entry: EventLogEntry = {
    id: `${event.session_id}-${event.turn_id}-${event.generation_id}-${event.seq}-${event.kind}-${receivedAt}`,
    event,
    firstEmittedAt: emittedAt,
    firstReceivedAt: receivedAt,
    emittedAt,
    receivedAt,
    count: 1,
    firstSeq: event.seq,
    lastSeq: event.seq,
  };
  return [entry, ...entries].slice(0, EVENT_LOG_LIMIT);
}

function canMergeEventLogEntry(entry: EventLogEntry, event: StreamEnvelope) {
  if (!isAggregatedEvent(event) || entry.event.kind !== event.kind) {
    return false;
  }
  return (
    entry.event.session_id === event.session_id &&
    entry.event.turn_id === event.turn_id &&
    entry.event.generation_id === event.generation_id &&
    entry.event.branch_state === event.branch_state
  );
}

function isAggregatedEvent(event: StreamEnvelope) {
  return AGGREGATED_EVENT_KINDS.has(event.kind);
}

function eventSummary(entry: EventLogEntry) {
  const payload = entry.event.payload as Record<string, unknown>;
  const state = typeof payload.state === "string" ? payload.state : entry.event.branch_state;
  const text = typeof payload.text === "string" ? payload.text.trim() : "";
  const pts = typeof entry.event.pts_ms === "number" ? `pts ${Math.round(entry.event.pts_ms)} ms` : "";
  const repeat = entry.count > 1 ? `x${entry.count}` : "";
  const lag = deliveryLagLabel(entry);
  const details = [state, text, pts, repeat, lag].filter(Boolean);
  return details.join(" · ");
}

function eventSequenceLabel(entry: EventLogEntry) {
  const seq =
    entry.firstSeq === entry.lastSeq ? `seq ${entry.lastSeq}` : `seq ${entry.firstSeq}-${entry.lastSeq}`;
  return `gen ${entry.event.generation_id} / ${seq}`;
}

function formatClock(value: number) {
  const date = new Date(value);
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  const millis = date.getMilliseconds().toString().padStart(3, "0");
  return `${hours}:${minutes}:${seconds}.${millis}`;
}

function msValue(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)} ms` : "--";
}

function deliveryLagLabel(entry: EventLogEntry) {
  if (typeof entry.event.emitted_at_ms !== "number") {
    return "";
  }
  const lagMs = entry.receivedAt - entry.emittedAt;
  if (!Number.isFinite(lagMs) || Math.abs(lagMs) > 5 * 60 * 1000) {
    return "server time";
  }
  return `rx +${Math.max(0, Math.round(lagMs))} ms`;
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${AGENT_BASE_URL}${path}`);
  return parseApiResponse<T>(response);
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${AGENT_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseApiResponse<T>(response);
}

async function parseApiResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    if (typeof detail === "string") {
      throw new Error(detail);
    }
    if (detail?.message) {
      throw new Error(detail.message);
    }
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return payload as T;
}

function runtimeChecks(runtime: RuntimeStatus | null): RuntimeCheck[] {
  return [
    { label: "Agent API", ok: runtime !== null, detail: runtime ? runtime.mode : "offline" },
    { label: "Qwen LLM artifact", ok: runtime?.artifacts.llm?.ok, detail: sizeDetail(runtime?.artifacts.llm) },
    { label: "GigaAM STT artifact", ok: runtime?.artifacts.stt?.ok, detail: sizeDetail(runtime?.artifacts.stt) },
    { label: "Qwen TTS artifact", ok: runtime?.artifacts.tts?.ok, detail: sizeDetail(runtime?.artifacts.tts) },
    { label: "Audio2Face artifact", ok: runtime?.artifacts.a2f?.ok, detail: sizeDetail(runtime?.artifacts.a2f) },
    { label: "VRM avatar", ok: runtime?.artifacts.avatar?.ok, detail: runtime?.avatar_url ?? "missing" },
    {
      label: "vLLM server",
      ok: runtime?.services.vllm?.ok,
      detail: runtime?.services.vllm?.base_url ?? "http://127.0.0.1:46111/v1",
    },
    { label: "GPU headroom", ok: runtime?.services.gpu?.ok, detail: gpuDetail(runtime?.services.gpu) },
    {
      label: "LiveKit server",
      ok: runtime?.services.livekit_server?.ok,
      detail: serviceEndpoint(runtime?.services.livekit_server),
    },
    {
      label: "LiveKit credentials",
      ok: runtime?.services.livekit_credentials?.ok,
      detail: runtime?.services.livekit_credentials?.detail ?? "unknown",
    },
    {
      label: "Voice pipeline",
      ok: runtime?.services.voice_pipeline?.ok,
      detail: runtime?.services.voice_pipeline?.detail ?? "not implemented",
    },
  ];
}

function toErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function latestVoiceLine(lines: VoiceLine[], role: VoiceLine["role"]) {
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (line.role === role && line.text.trim().length > 0) {
      return line;
    }
  }
  return null;
}

function shortId(value: string) {
  return value.slice(0, 8);
}

function sizeDetail(item: RuntimeStatus["artifacts"][string] | undefined) {
  if (!item) {
    return "unknown";
  }
  if (!item.ok) {
    return "missing";
  }
  return typeof item.size_gb === "number" ? `${item.size_gb} GB` : "present";
}

function serviceEndpoint(item: RuntimeStatus["services"][string] | undefined) {
  if (!item) {
    return "unknown";
  }
  if (item.host && item.port) {
    return `${item.host}:${item.port} (${item.detail ?? "unknown"})`;
  }
  return item.detail ?? "unknown";
}

function gpuDetail(item: RuntimeStatus["services"][string] | undefined) {
  if (!item) {
    return "unknown";
  }
  const free = typeof item.free_gb === "number" ? `${item.free_gb} GB free` : (item.detail ?? "unknown");
  const min = typeof item.min_free_gb === "number" ? `min ${item.min_free_gb} GB` : "min unknown";
  return `${free} / ${min}`;
}

function voiceUnavailableReason(runtime: RuntimeStatus | null) {
  if (!runtime) {
    return "runtime status unavailable";
  }
  if (!runtime.ready.all_artifacts) {
    return "required artifacts are missing";
  }
  if (!runtime.services.vllm?.ok) {
    return `vLLM ${runtime.services.vllm?.detail ?? "offline"}`;
  }
  if (!runtime.services.gpu?.ok) {
    return `GPU headroom ${gpuDetail(runtime.services.gpu)}`;
  }
  if (!runtime.services.livekit_credentials?.ok) {
    return "LiveKit credentials missing";
  }
  if (!runtime.services.livekit_server?.ok) {
    return `LiveKit ${serviceEndpoint(runtime.services.livekit_server)}`;
  }
  if (!runtime.services.voice_pipeline?.ok) {
    return runtime.services.voice_pipeline?.detail ?? "voice pipeline not ready";
  }
  return "not ready";
}

createRoot(document.getElementById("root")!).render(<App />);
