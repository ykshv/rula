import type { AvatarState } from "./protocol";

type SetMicrophoneEnabled = (enabled: boolean) => Promise<void>;
type ReportError = (message: string) => void;

const DEFAULT_RELEASE_DELAY_MS = 220;
const MOBILE_USER_AGENT = /Android|iPhone|iPad|iPod|Mobile/i;

export interface MobileEchoSafeMicGateOptions {
  enabled?: boolean;
  releaseDelayMs?: number;
}

export class MobileEchoSafeMicGate {
  private releaseTimer = 0;
  private micEnabled = true;
  private operation: Promise<void> = Promise.resolve();

  constructor(
    private readonly setMicrophoneEnabled: SetMicrophoneEnabled,
    private readonly reportError: ReportError,
    private readonly options: MobileEchoSafeMicGateOptions = {},
  ) {}

  get enabled(): boolean {
    return this.options.enabled ?? shouldUseMobileEchoSafeMode();
  }

  onAvatarState(state: AvatarState): void {
    if (!this.enabled) {
      return;
    }
    if (state === "speaking") {
      this.clearReleaseTimer();
      this.setEnabled(false);
      return;
    }
    if (state === "idle" || state === "listening" || state === "interrupted") {
      this.scheduleRelease();
    }
  }

  reset(): void {
    this.clearReleaseTimer();
    this.micEnabled = true;
    this.operation = Promise.resolve();
  }

  private scheduleRelease(): void {
    this.clearReleaseTimer();
    const delayMs = this.options.releaseDelayMs ?? DEFAULT_RELEASE_DELAY_MS;
    this.releaseTimer = window.setTimeout(() => {
      this.releaseTimer = 0;
      this.setEnabled(true);
    }, delayMs);
  }

  private setEnabled(enabled: boolean): void {
    if (this.micEnabled === enabled) {
      return;
    }
    this.micEnabled = enabled;
    this.operation = this.operation
      .catch(() => undefined)
      .then(async () => {
        await this.setMicrophoneEnabled(enabled);
      })
      .catch((error: unknown) => {
        this.micEnabled = !enabled;
        const message = error instanceof Error ? error.message : String(error);
        this.reportError(`Microphone gate failed: ${message}`);
      });
  }

  private clearReleaseTimer(): void {
    if (this.releaseTimer !== 0) {
      window.clearTimeout(this.releaseTimer);
      this.releaseTimer = 0;
    }
  }
}

export function shouldUseMobileEchoSafeMode(): boolean {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return false;
  }
  const coarsePointer =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;
  const touchDevice = navigator.maxTouchPoints > 0;
  return MOBILE_USER_AGENT.test(navigator.userAgent) || (coarsePointer && touchDevice);
}
