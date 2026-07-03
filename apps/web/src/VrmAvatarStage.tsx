import React from "react";
import {
  AmbientLight,
  Clock,
  Color,
  DirectionalLight,
  Euler,
  Object3D,
  PerspectiveCamera,
  Quaternion,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { arkitToVrm } from "./voice/faceMapping";
import type { FaceScheduler } from "./voice/FaceScheduler";

const DEFAULT_AVATAR_URL = "/assets/avatars/default-female/AliciaSolid_vrm-0.51.vrm";

export type AvatarStatus =
  | "offline"
  | "ready"
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "interrupted";
type LoadState = "loading" | "ready" | "error";
type GazeOffset = { x: number; y: number };
type PointerGazeSample = GazeOffset & { influence: number };

const POINTER_GAZE_LIMIT = { x: 0.62, y: 0.42 };
const POINTER_DEAD_ZONE = 0.035;
const DESKTOP_CAMERA = {
  position: { x: 0, y: 1.35, z: 2.35 },
  target: { x: 0, y: 1.28, z: 0 },
};
const MOBILE_PORTRAIT_CAMERA = {
  position: { x: 0, y: 1.54, z: 1.35 },
  target: { x: 0, y: 1.47, z: 0 },
};

/**
 * Procedural idle life: blinks, gaze wander, breathing and head sway keep the
 * avatar alive between turns; while speaking, scheduled Audio2Face frames own
 * the mouth and eye channels.
 */
class IdleBehaviour {
  private nextBlinkAt = 1.5;
  private blinkPhase = -1;
  private gazeTarget = { x: 0, y: 0 };
  private gaze = { x: 0, y: 0 };
  private nextGazeAt = 2;
  private nodPhase = -1;
  private nextNodAt = 1.2;
  time = 0;

  update(delta: number, status: AvatarStatus) {
    this.time += delta;
  }

  blink(delta: number): number {
    if (this.blinkPhase >= 0) {
      this.blinkPhase += delta;
      const duration = 0.14;
      if (this.blinkPhase >= duration) {
        this.blinkPhase = -1;
        return 0;
      }
      const t = this.blinkPhase / duration;
      return t < 0.5 ? t * 2 : (1 - t) * 2;
    }
    if (this.time >= this.nextBlinkAt) {
      this.blinkPhase = 0;
      this.nextBlinkAt = this.time + 1.8 + Math.random() * 4.2;
      return 0;
    }
    return 0;
  }

  gazeOffsets(delta: number, status: AvatarStatus): GazeOffset {
    if (this.time >= this.nextGazeAt) {
      const magnitude = status === "thinking" ? 0.5 : 0.22;
      this.gazeTarget = {
        x: (Math.random() - 0.5) * 2 * magnitude,
        y: (Math.random() - 0.35) * (status === "thinking" ? 0.5 : 0.18),
      };
      this.nextGazeAt = this.time + 1.2 + Math.random() * 2.6;
    }
    const k = Math.min(1, delta * 6);
    this.gaze.x += (this.gazeTarget.x - this.gaze.x) * k;
    this.gaze.y += (this.gazeTarget.y - this.gaze.y) * k;
    return this.gaze;
  }

  /** Small affirmative nod while listening. */
  nod(delta: number, status: AvatarStatus): number {
    if (status !== "listening") {
      this.nodPhase = -1;
      return 0;
    }
    if (this.nodPhase >= 0) {
      this.nodPhase += delta;
      const duration = 0.55;
      if (this.nodPhase >= duration) {
        this.nodPhase = -1;
        this.nextNodAt = this.time + 2.2 + Math.random() * 3.4;
        return 0;
      }
      return Math.sin((this.nodPhase / duration) * Math.PI) * 0.07;
    }
    if (this.time >= this.nextNodAt) {
      this.nodPhase = 0;
    }
    return 0;
  }
}

class PointerAttention {
  private target: GazeOffset = { x: 0, y: 0 };
  private current: GazeOffset = { x: 0, y: 0 };
  private influence = 0;
  private targetInfluence = 0;

  track(event: PointerEvent, element: HTMLElement) {
    if (event.pointerType === "touch") {
      this.release();
      return;
    }

    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return;
    }

    const normalizedX = clamp(((event.clientX - rect.left) / rect.width) * 2 - 1, -1, 1);
    const normalizedY = clamp(((event.clientY - rect.top) / rect.height) * 2 - 1, -1, 1);

    this.target.x = softenPointerAxis(normalizedX) * POINTER_GAZE_LIMIT.x;
    this.target.y = -softenPointerAxis(normalizedY) * POINTER_GAZE_LIMIT.y;
    this.targetInfluence = 1;
  }

  release() {
    this.target.x = 0;
    this.target.y = 0;
    this.targetInfluence = 0;
  }

  sample(delta: number): PointerGazeSample {
    const gazeK = dampedStep(delta, 7.5);
    const influenceK = dampedStep(delta, this.targetInfluence > this.influence ? 6 : 3.2);

    this.current.x += (this.target.x - this.current.x) * gazeK;
    this.current.y += (this.target.y - this.current.y) * gazeK;
    this.influence += (this.targetInfluence - this.influence) * influenceK;

    if (this.targetInfluence === 0 && this.influence < 0.002) {
      this.influence = 0;
    }

    return { x: this.current.x, y: this.current.y, influence: this.influence };
  }
}

export function VrmAvatarStage({
  faceScheduler,
  status,
  avatarUrl = DEFAULT_AVATAR_URL,
  subtitle,
}: {
  faceScheduler: FaceScheduler | null;
  status: AvatarStatus;
  avatarUrl?: string;
  subtitle?: string | null;
}) {
  const stageRef = React.useRef<HTMLDivElement | null>(null);
  const mountRef = React.useRef<HTMLDivElement | null>(null);
  const vrmRef = React.useRef<VRM | null>(null);
  const statusRef = React.useRef<AvatarStatus>(status);
  const schedulerRef = React.useRef<FaceScheduler | null>(faceScheduler);
  const [loadState, setLoadState] = React.useState<LoadState>("loading");
  const [error, setError] = React.useState<string | null>(null);

  statusRef.current = status;
  schedulerRef.current = faceScheduler;

  React.useEffect(() => {
    const stage = stageRef.current;
    const mount = mountRef.current;
    if (!stage || !mount) {
      return;
    }

    let disposed = false;
    let animationFrame = 0;
    const clock = new Clock(false);
    const idle = new IdleBehaviour();
    const pointer = new PointerAttention();

    const scene = new Scene();
    scene.background = new Color("#0c0f0e");

    const camera = new PerspectiveCamera(28, 1, 0.1, 20);
    applyCameraPreset(camera, DESKTOP_CAMERA);

    const renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.debug.checkShaderErrors = import.meta.env.DEV;
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.domElement.className = "avatar-canvas";
    mount.appendChild(renderer.domElement);

    const ambient = new AmbientLight("#f4f2ea", 1.2);
    const key = new DirectionalLight("#fff6df", 2.8);
    key.position.set(1.2, 2.2, 2.5);
    const fill = new DirectionalLight("#b2dfcb", 1.4);
    fill.position.set(-2.0, 1.4, 1.6);
    scene.add(ambient, key, fill);

    const resize = () => {
      const { width, height } = mount.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(width));
      const nextHeight = Math.max(1, Math.floor(height));
      const isNarrow = nextWidth < 720;
      applyCameraPreset(camera, isNarrow ? MOBILE_PORTRAIT_CAMERA : DESKTOP_CAMERA);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight, false);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    const trackPointer = (event: PointerEvent) => {
      pointer.track(event, stage);
    };
    const releasePointer = () => {
      pointer.release();
    };
    stage.addEventListener("pointerenter", trackPointer, { passive: true });
    stage.addEventListener("pointermove", trackPointer, { passive: true });
    stage.addEventListener("pointerleave", releasePointer, { passive: true });
    stage.addEventListener("pointercancel", releasePointer, { passive: true });
    window.addEventListener("blur", releasePointer);

    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    setLoadState("loading");
    setError(null);

    const animateFrame = (delta: number) => {
      const vrm = vrmRef.current;
      if (!vrm) {
        return;
      }
      const currentStatus = statusRef.current;
      idle.update(delta, currentStatus);

      const manager = vrm.expressionManager;
      const scheduler = schedulerRef.current;
      const speech = scheduler ? scheduler.sample(performance.now()) : null;

      const blink = idle.blink(delta);
      const gaze = blendGaze(idle.gazeOffsets(delta, currentStatus), pointer.sample(delta));
      const nod = idle.nod(delta, currentStatus);

      if (manager) {
        if (speech) {
          const weights = arkitToVrm(speech);
          setExpressionAliases(manager, ["aa", "a"], weights.aa);
          setExpressionAliases(manager, ["ih", "i"], weights.ih);
          setExpressionAliases(manager, ["ou", "u"], weights.ou);
          setExpressionAliases(manager, ["ee", "e"], weights.ee);
          setExpressionAliases(manager, ["oh", "o"], weights.oh);
          setExpressionAliases(manager, ["happy", "joy", "fun"], weights.happy);
          setExpressionAliases(manager, ["blinkLeft", "blink_l"], Math.max(weights.blinkLeft, blink));
          setExpressionAliases(manager, ["blinkRight", "blink_r"], Math.max(weights.blinkRight, blink));
          setExpressionAliases(manager, ["blink"], Math.max(weights.blinkLeft, weights.blinkRight, blink));
        } else {
          setExpressionAliases(manager, ["aa", "a"], 0);
          setExpressionAliases(manager, ["ih", "i"], 0);
          setExpressionAliases(manager, ["ou", "u"], 0);
          setExpressionAliases(manager, ["ee", "e"], 0);
          setExpressionAliases(manager, ["oh", "o"], 0);
          setExpressionAliases(manager, ["happy", "joy", "fun"], currentStatus === "speaking" ? 0.1 : 0.06);
          setExpressionAliases(manager, ["blinkLeft", "blink_l"], blink);
          setExpressionAliases(manager, ["blinkRight", "blink_r"], blink);
          setExpressionAliases(manager, ["blink"], blink);
        }
        manager.update();
      }

      const head = vrm.humanoid.getNormalizedBoneNode("head");
      const neck = vrm.humanoid.getNormalizedBoneNode("neck");
      const chest = vrm.humanoid.getNormalizedBoneNode("chest");
      const breathing = Math.sin(idle.time * 1.4) * 0.012;
      const sway = Math.sin(idle.time * 0.6) * 0.02;
      if (head) {
        head.rotation.set(gaze.y * 0.35 + nod + breathing * 0.4, gaze.x * 0.5 + sway, 0);
      }
      if (neck) {
        neck.rotation.set(nod * 0.5, gaze.x * 0.12 + sway * 0.6, 0);
      }
      if (chest) {
        chest.rotation.set(0.02 + breathing, 0, 0);
      }

      const lookAt = vrm.lookAt;
      if (lookAt) {
        lookAt.yaw = gaze.x * 12;
        lookAt.pitch = gaze.y * 10;
      }
    };

    let lastRenderMs = 0;
    const minFrameMs = 1000 / 30;
    const render = (nowMs = 0) => {
      animationFrame = window.requestAnimationFrame(render);
      if (nowMs - lastRenderMs < minFrameMs) {
        return;
      }
      lastRenderMs = nowMs;
      const delta = clock.getDelta();
      animateFrame(delta);
      vrmRef.current?.update(delta);
      renderer.render(scene, camera);
    };

    const startRenderLoop = () => {
      if (animationFrame !== 0) {
        return;
      }
      clock.start();
      render();
    };

    loader.load(
      avatarUrl,
      (gltf) => {
        if (disposed) {
          return;
        }

        const vrm = gltf.userData.vrm as VRM | undefined;
        if (!vrm) {
          setLoadState("error");
          setError("VRM loader did not return a model");
          return;
        }

        VRMUtils.rotateVRM0(vrm);
        vrm.scene.position.set(0, 0, 0);
        vrm.scene.rotation.y = Math.PI;
        applyDefaultPose(vrm);
        scene.add(vrm.scene);
        vrmRef.current = vrm;
        renderer
          .compileAsync(scene, camera)
          .then(() => {
            if (disposed) {
              return;
            }
            setLoadState("ready");
            startRenderLoop();
          })
          .catch((compileError: unknown) => {
            if (disposed) {
              return;
            }
            setLoadState("error");
            setError(compileError instanceof Error ? compileError.message : "Cannot compile VRM materials");
          });
      },
      undefined,
      (loadError) => {
        if (disposed) {
          return;
        }
        setLoadState("error");
        setError(loadError instanceof Error ? loadError.message : "Cannot load default VRM avatar");
      },
    );

    return () => {
      disposed = true;
      if (animationFrame !== 0) {
        window.cancelAnimationFrame(animationFrame);
      }
      resizeObserver.disconnect();
      stage.removeEventListener("pointerenter", trackPointer);
      stage.removeEventListener("pointermove", trackPointer);
      stage.removeEventListener("pointerleave", releasePointer);
      stage.removeEventListener("pointercancel", releasePointer);
      window.removeEventListener("blur", releasePointer);
      vrmRef.current = null;
      renderer.dispose();
      disposeObject(scene);
      renderer.domElement.remove();
    };
  }, [avatarUrl]);

  return (
    <div ref={stageRef} className="avatar-stage" data-status={status}>
      <div ref={mountRef} className="vrm-stage" aria-label="VRM avatar viewport" />
      {loadState !== "ready" && (
        <div className="asset-state" role="status">
          <strong>{loadState === "loading" ? "Loading VRM avatar" : "Avatar asset unavailable"}</strong>
          {error && <span>{error}</span>}
        </div>
      )}
      {subtitle && (
        <div className="avatar-subtitle" aria-live="polite">
          <p>{subtitle}</p>
        </div>
      )}
    </div>
  );
}

function blendGaze(idle: GazeOffset, pointer: PointerGazeSample): GazeOffset {
  const idleWeight = 1 - pointer.influence;
  return {
    x: idle.x * idleWeight + pointer.x * pointer.influence,
    y: idle.y * idleWeight + pointer.y * pointer.influence,
  };
}

function softenPointerAxis(value: number): number {
  const absValue = Math.abs(value);
  if (absValue <= POINTER_DEAD_ZONE) {
    return 0;
  }
  const normalized = (absValue - POINTER_DEAD_ZONE) / (1 - POINTER_DEAD_ZONE);
  return Math.sign(value) * Math.sin((normalized * Math.PI) / 2);
}

function dampedStep(delta: number, speed: number): number {
  return 1 - Math.exp(-delta * speed);
}

function applyCameraPreset(
  camera: PerspectiveCamera,
  preset: {
    position: { x: number; y: number; z: number };
    target: { x: number; y: number; z: number };
  },
) {
  camera.position.set(preset.position.x, preset.position.y, preset.position.z);
  camera.lookAt(preset.target.x, preset.target.y, preset.target.z);
}

function setExpression(manager: NonNullable<VRM["expressionManager"]>, name: string, value: number) {
  if (manager.getExpression(name)) {
    manager.setValue(name, Math.min(1, Math.max(0, value)));
  }
}

function setExpressionAliases(
  manager: NonNullable<VRM["expressionManager"]>,
  names: string[],
  value: number,
) {
  for (const name of names) {
    setExpression(manager, name, value);
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function applyDefaultPose(vrm: VRM) {
  vrm.humanoid.setNormalizedPose({
    leftUpperArm: { rotation: quaternionFromEuler(0, 0, 1.1) },
    rightUpperArm: { rotation: quaternionFromEuler(0, 0, -1.1) },
    leftLowerArm: { rotation: quaternionFromEuler(0, 0, 0.22) },
    rightLowerArm: { rotation: quaternionFromEuler(0, 0, -0.22) },
    spine: { rotation: quaternionFromEuler(0.04, 0, 0) },
    chest: { rotation: quaternionFromEuler(0.02, 0, 0) },
  });
  vrm.humanoid.update();
}

function quaternionFromEuler(x: number, y: number, z: number): [number, number, number, number] {
  return new Quaternion().setFromEuler(new Euler(x, y, z)).toArray() as [number, number, number, number];
}

function disposeObject(root: Object3D) {
  root.traverse((object) => {
    const candidate = object as Object3D & {
      geometry?: { dispose: () => void };
      material?: { dispose: () => void } | Array<{ dispose: () => void }>;
    };
    candidate.geometry?.dispose();
    if (Array.isArray(candidate.material)) {
      for (const material of candidate.material) {
        material.dispose();
      }
    } else {
      candidate.material?.dispose();
    }
  });
}
