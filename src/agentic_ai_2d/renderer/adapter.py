"""Renderer interface plus a deterministic local FFmpeg implementation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, sqrt
from pathlib import Path
import subprocess
import time
from typing import Iterable, Protocol

from PIL import Image

from ..models.storyboard import MotionKind
from ..models.timeline import Timeline, VisualClip, VisualTrack, VisualTrackKind
from ..models.viseme_timeline import MouthShape, VisemeTimeline
from ..storage import ArtifactManager


@dataclass(frozen=True, slots=True)
class RenderResult:
    output_path: Path
    manifest_path: Path
    output_digest: str


class RendererAdapter(Protocol):
    def render(self, timeline: Timeline) -> RenderResult:
        """Render a validated Timeline and return immutable output references."""


class SimulatedRemotionRenderer:
    """Writes deterministic placeholder output until a real renderer is attached."""

    def __init__(self, output_root: Path | str) -> None:
        self._output_root = Path(output_root)

    def render(self, timeline: Timeline) -> RenderResult:
        render_dir = self._output_root / timeline.project_id / f"v{timeline.project_version}"
        render_dir.mkdir(parents=True, exist_ok=True)
        output_path = render_dir / "output.mp4"
        manifest_path = render_dir / "render_manifest.json"
        timeline_json = json.dumps(
            timeline.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        output = b"\x00\x00\x00\x18ftypmp42isom" + sha256(timeline_json).digest()
        output_path.write_bytes(output)
        output_digest = sha256(output).hexdigest()
        manifest = {
            "renderer": "simulated_remotion",
            "timeline_id": timeline.timeline_id,
            "project_id": timeline.project_id,
            "project_version": timeline.project_version,
            "timeline_digest": sha256(timeline_json).hexdigest(),
            "output_file": output_path.name,
            "output_digest": output_digest,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        return RenderResult(output_path=output_path, manifest_path=manifest_path, output_digest=output_digest)


class LocalFfmpegRenderer:
    """Render a Timeline directly with FFmpeg while preserving Phase 1 contracts."""

    _DEFAULT_BACKGROUND = "#d8edf2"
    _MOUTH_SHAPE_ORDER = [
        MouthShape.A,
        MouthShape.B,
        MouthShape.C,
        MouthShape.D,
        MouthShape.E,
        MouthShape.F,
        MouthShape.G,
        MouthShape.H,
        MouthShape.X,
    ]

    def __init__(
        self,
        storage: ArtifactManager,
        *,
        output_root: Path | str,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        timeout_seconds: int = 900,
        visemes: Iterable[VisemeTimeline] = (),
    ) -> None:
        self._storage = storage
        self._output_root = Path(output_root).resolve()
        self._ffmpeg_binary = ffmpeg_binary
        self._ffprobe_binary = ffprobe_binary
        self._timeout_seconds = timeout_seconds
        self._visemes = {viseme.viseme_timeline_id: viseme for viseme in visemes}

    def render(self, timeline: Timeline) -> RenderResult:
        render_dir = (self._output_root / timeline.project_id / f"v{timeline.project_version}").resolve()
        inputs_dir = render_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = render_dir / "output.mp4"
        manifest_path = render_dir / "render_manifest.json"
        timeline_path = inputs_dir / "timeline.json"
        asset_map_path = inputs_dir / "asset-map.json"
        render_config_path = inputs_dir / "render-config.json"
        timeline_payload = timeline.model_dump(mode="json")
        timeline_json = json.dumps(timeline_payload, sort_keys=True, separators=(",", ":"))
        timeline_path.write_text(timeline_json, encoding="utf-8")
        asset_map = {
            "assets": self._asset_map(timeline),
            "visemes": {key: value.model_dump(mode="json") for key, value in self._visemes.items()},
        }
        asset_map_path.write_text(json.dumps(asset_map, sort_keys=True, indent=2), encoding="utf-8")
        render_config = {
            "renderer": "local_ffmpeg",
            "ffmpeg_binary": self._ffmpeg_binary,
            "ffprobe_binary": self._ffprobe_binary,
            "timeout_seconds": self._timeout_seconds,
            "canvas": {
                "width": timeline.width,
                "height": timeline.height,
                "fps": timeline.fps,
                "duration_frames": timeline.duration_frames,
            },
        }
        render_config_path.write_text(json.dumps(render_config, sort_keys=True, indent=2), encoding="utf-8")

        asset_sizes = self._asset_sizes(asset_map["assets"])
        command = self._build_ffmpeg_command(timeline, asset_map["assets"], asset_sizes, output_path)
        started_at = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        if completed.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Local FFmpeg render failed: {completed.stderr}")
        media_info = self._validate_mp4(output_path)
        output_digest = sha256(output_path.read_bytes()).hexdigest()
        manifest = {
            "renderer": "local_ffmpeg",
            "timeline_id": timeline.timeline_id,
            "project_id": timeline.project_id,
            "project_version": timeline.project_version,
            "timeline_digest": sha256(timeline_json.encode("utf-8")).hexdigest(),
            "output_file": output_path.name,
            "output_digest": output_digest,
            "asset_digests": {
                asset_id: self._storage.get_metadata(asset_id).digest
                for asset_id in asset_map["assets"]
            },
            "ffmpeg_version": self._binary_version(self._ffmpeg_binary),
            "ffprobe_version": self._binary_version(self._ffprobe_binary),
            "elapsed_seconds": elapsed_seconds,
            "media_info": media_info,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
        return RenderResult(output_path=output_path, manifest_path=manifest_path, output_digest=output_digest)

    def _asset_map(self, timeline: Timeline) -> dict[str, str]:
        asset_ids = {track.asset_id for track in timeline.audio_tracks}
        asset_ids.update(clip.asset_id for track in timeline.visual_tracks for clip in track.clips)
        return {
            asset_id: str(self._storage.absolute_path(self._storage.get_metadata(asset_id)).resolve())
            for asset_id in sorted(asset_ids)
        }

    @staticmethod
    def _asset_sizes(asset_map: dict[str, str]) -> dict[str, tuple[int, int]]:
        image_sizes: dict[str, tuple[int, int]] = {}
        for asset_id, asset_path in asset_map.items():
            if Path(asset_path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            with Image.open(asset_path) as image:
                image_sizes[asset_id] = image.size
        return image_sizes

    def _build_ffmpeg_command(
        self,
        timeline: Timeline,
        asset_map: dict[str, str],
        asset_sizes: dict[str, tuple[int, int]],
        output_path: Path,
    ) -> list[str]:
        duration_seconds = timeline.duration_frames / timeline.fps
        command = [
            self._ffmpeg_binary,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={self._DEFAULT_BACKGROUND}:s={timeline.width}x{timeline.height}:r={timeline.fps}:d={duration_seconds:.6f}",
        ]
        visual_asset_ids = sorted({clip.asset_id for track in timeline.visual_tracks for clip in track.clips})
        input_indexes: dict[str, int] = {}
        for asset_id in visual_asset_ids:
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(timeline.fps),
                    "-t",
                    f"{duration_seconds:.6f}",
                    "-i",
                    asset_map[asset_id],
                ]
            )
            input_indexes[asset_id] = len(input_indexes) + 1
        narration_asset_id = timeline.audio_tracks[0].asset_id
        command.extend(["-i", asset_map[narration_asset_id]])
        audio_input_index = len(visual_asset_ids) + 1
        filter_complex = self._build_filter_complex(timeline, input_indexes, asset_sizes)
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                f"{audio_input_index}:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ]
        )
        return command

    def _build_filter_complex(
        self,
        timeline: Timeline,
        input_indexes: dict[str, int],
        asset_sizes: dict[str, tuple[int, int]],
    ) -> str:
        filters = ["[0:v]format=rgba[base0]"]
        current = "base0"
        step = 0
        for track in sorted(timeline.visual_tracks, key=lambda item: item.z_index):
            if track.kind is VisualTrackKind.CHARACTER_MOUTH:
                current, step = self._append_mouth_track(filters, current, step, timeline, track, input_indexes)
                continue
            for clip in track.clips:
                current, step = self._append_visual_clip(
                    filters, current, step, timeline, track, clip, input_indexes, asset_sizes
                )
        filters.append(f"[{current}]format=yuv420p[vout]")
        return ";".join(filters)

    def _append_visual_clip(
        self,
        filters: list[str],
        current: str,
        step: int,
        timeline: Timeline,
        track: VisualTrack,
        clip: VisualClip,
        input_indexes: dict[str, int],
        asset_sizes: dict[str, tuple[int, int]],
    ) -> tuple[str, int]:
        source_label = f"[{input_indexes[clip.asset_id]}:v]"
        prepared_label = f"layer_{step}"
        source_width, source_height = asset_sizes[clip.asset_id]
        if track.kind is VisualTrackKind.BACKGROUND:
            filters.append(f"{source_label}scale={timeline.width}:{timeline.height},format=rgba[{prepared_label}]")
            x_expr = "0"
            y_expr = "0"
        else:
            base_width = max(1, int(round(timeline.width * 0.35 * (clip.scale or 1.0))))
            base_height = max(1, int(round(base_width * source_height / source_width)))
            filters.append(self._prepared_visual_chain(source_label, prepared_label, clip, base_width, base_height))
            x_expr, y_expr = self._overlay_position(timeline, clip)
        next_label = f"comp_{step}"
        # Keep contact coordinates at full pixel resolution until final encoding.
        contact = clip.motion is not None and clip.motion.landing_surface_y_percent is not None
        overlay_format = ':format=rgb' if contact else ''
        filters.append(
            f"[{current}][{prepared_label}]overlay="
            f"x='{x_expr}':y='{y_expr}':eval=frame{overlay_format}:enable='between(n,{clip.start_frame},{clip.start_frame + clip.duration_frames - 1})'"
            f"[{next_label}]"
        )
        return next_label, step + 1

    def _append_mouth_track(
        self,
        filters: list[str],
        current: str,
        step: int,
        timeline: Timeline,
        track: VisualTrack,
        input_indexes: dict[str, int],
    ) -> tuple[str, int]:
        mouth_width = max(1, int(round(timeline.width * 0.12)))
        mouth_height = max(1, int(round(timeline.height * 0.12)))
        for clip in track.clips:
            if clip.viseme_timeline_id is None:
                continue
            viseme = self._visemes.get(clip.viseme_timeline_id)
            if viseme is None:
                raise KeyError(f"missing viseme timeline {clip.viseme_timeline_id!r} for mouth track")
            x_expr, y_expr = self._overlay_position(timeline, clip)
            input_label = f"[{input_indexes[clip.asset_id]}:v]"
            for cue in viseme.cues:
                shape_index = self._MOUTH_SHAPE_ORDER.index(cue.mouth_shape)
                start_frame = max(clip.start_frame, cue.start_frame)
                end_frame = min(clip.start_frame + clip.duration_frames - 1, cue.end_frame)
                if end_frame < start_frame:
                    continue
                prepared_label = f"mouth_{step}"
                filters.append(
                    f"{input_label}crop=w=iw:h=ih/9:x=0:y={shape_index}*ih/9,"
                    f"scale={mouth_width}:{mouth_height}:flags=lanczos,format=rgba[{prepared_label}]"
                )
                next_label = f"comp_{step}"
                filters.append(
                    f"[{current}][{prepared_label}]overlay="
                    f"x='{x_expr}':y='{y_expr}':eval=frame:enable='between(n,{start_frame},{end_frame})'"
                    f"[{next_label}]"
                )
                current = next_label
                step += 1
        return current, step

    def _prepared_visual_chain(
        self,
        source_label: str,
        prepared_label: str,
        clip: VisualClip,
        base_width: int,
        base_height: int,
    ) -> str:
        motion = clip.motion
        has_rotation = clip.rotation_degrees != 0 or (motion is not None and motion.kind is MotionKind.FLOWER_SWAY)
        if motion is not None and motion.landing_surface_y_percent is not None:
            # The aligned rig translates a shared sprite canvas without rotation padding.
            return f"{source_label}scale={base_width}:{base_height}:flags=lanczos,format=rgba[{prepared_label}]"
        if not has_rotation and (motion is None or motion.kind is MotionKind.STATIC):
            return f"{source_label}scale={base_width}:{base_height}:flags=lanczos,format=rgba[{prepared_label}]"
        local_frame = f"(n-{clip.start_frame})"
        if motion.kind is MotionKind.BUTTERFLY_FLAP:
            amplitude = motion.amplitude_percent / 100.0
            flap = f"(1-{amplitude:.6f}*(0.5+0.5*sin(({local_frame}/{motion.period_frames})*2*PI)))"
            # Pad around the wing root first, making the requested pivot the layer center.
            anchor_x = base_width * clip.anchor_x_percent / 100.0
            anchor_y = base_height * clip.anchor_y_percent / 100.0
            padded_width = int(ceil(2 * max(anchor_x, base_width - anchor_x)))
            padded_height = int(ceil(2 * max(anchor_y, base_height - anchor_y)))
            pad_x = int(round(padded_width / 2 - anchor_x))
            pad_y = int(round(padded_height / 2 - anchor_y))
            return (
                f"{source_label}scale={base_width}:{base_height}:flags=lanczos,"
                f"pad={padded_width}:{padded_height}:{pad_x}:{pad_y}:color=0x00000000,"
                f"scale=w='max(1,iw*{flap})':h=ih:eval=frame:flags=lanczos,"
                f"format=rgba[{prepared_label}]"
            )
        sway = "0"
        if motion is not None and motion.kind is MotionKind.FLOWER_SWAY:
            sway = f"{motion.amplitude_percent:.6f}*sin(({local_frame}/{motion.period_frames})*2*PI)"
        angle = f"(({clip.rotation_degrees:.6f}+{sway})*PI/180)"
        # Pad around the requested pivot so FFmpeg's center rotation preserves that pivot.
        anchor_x = base_width * clip.anchor_x_percent / 100.0
        anchor_y = base_height * clip.anchor_y_percent / 100.0
        padded_width = int(ceil(2 * max(anchor_x, base_width - anchor_x)))
        padded_height = int(ceil(2 * max(anchor_y, base_height - anchor_y)))
        pad_x = int(round(padded_width / 2 - anchor_x))
        pad_y = int(round(padded_height / 2 - anchor_y))
        return (
            f"{source_label}scale={base_width}:{base_height}:flags=lanczos,"
            f"pad={padded_width}:{padded_height}:{pad_x}:{pad_y}:color=0x00000000,"
            f"rotate=a='{angle}':ow=rotw(iw):oh=roth(ih):c=none,format=rgba[{prepared_label}]"
        )

    def _overlay_position(self, timeline: Timeline, clip: VisualClip) -> tuple[str, str]:
        motion = clip.motion
        x_percent = clip.x_percent if clip.x_percent is not None else 50.0
        y_percent = clip.y_percent if clip.y_percent is not None else 50.0
        x_expr = f"{timeline.width}*({x_percent:.6f}/100)-overlay_w/2"
        y_expr = f"{timeline.height}*({y_percent:.6f}/100)-overlay_h/2"
        if motion is None:
            return x_expr, y_expr
        if motion.kind is MotionKind.BIRD_HOP:
            start_x = motion.start_x_percent if motion.start_x_percent is not None else x_percent
            end_x = motion.end_x_percent if motion.end_x_percent is not None else x_percent
            # Each hop covers one fixed period. Horizontal travel is linear during flight;
            # vertical travel is a quadratic parabola that returns cleanly to the fence.
            hop_frames = min(clip.duration_frames - 1, motion.period_frames * motion.hop_count)
            progress = f"min(1,(n-{clip.start_frame})/{max(1, hop_frames)})"
            if motion.landing_surface_y_percent is not None:
                # Overlay n starts at 1; timestamps align the initial landing and SFX.
                progress = f"min(1,max(0,(t*{timeline.fps}-{clip.start_frame}))/{max(1, hop_frames)})"
            hop_phase = f"mod(({progress})*{motion.hop_count},1)"
            hop = f"4*({hop_phase})*(1-({hop_phase}))"
            x_expr = f"{timeline.width}*(({start_x:.6f}+({end_x:.6f}-{start_x:.6f})*({progress}))/100)-overlay_w/2"
            y_expr = f"{timeline.height}*(({y_percent:.6f}-{hop}*{motion.amplitude_percent:.6f})/100)-overlay_h/2"
            if motion.landing_surface_y_percent is not None and motion.foot_y_percent is not None:
                # Derive contact again at actual render resolution (including previews).
                y_expr = (f"{timeline.height}*(({motion.landing_surface_y_percent:.9f}"
                          f"-{hop}*{motion.amplitude_percent:.9f})/100)"
                          f"-overlay_h*{motion.foot_y_percent / 100:.12f}")
        return x_expr, y_expr

    def _validate_mp4(self, output_path: Path) -> dict[str, object]:
        completed = subprocess.run(
            [
                self._ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=index,codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe could not validate rendered MP4: {completed.stderr}")
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if not any(stream.get("codec_type") == "video" for stream in streams):
            raise RuntimeError("rendered MP4 does not contain a video stream")
        return payload

    @staticmethod
    def _binary_version(binary: str) -> str:
        completed = subprocess.run(
            [binary, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            return "unknown"
        return completed.stdout.splitlines()[0].strip() if completed.stdout else "unknown"


LocalRemotionRenderer = LocalFfmpegRenderer
