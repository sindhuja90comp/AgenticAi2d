# Agentic AI 2D

## What This Project Does

This project makes a short 2D animation from a supported prompt, such as a butterfly and a flower. It creates narration, mouth movement, simple motion, and an MP4 video.

## Pipeline in Simple Words

```text
+--------------------+------------------------------------------------+---------------------------+
| Unit               | What it does                                   | AI?                       |
+--------------------+------------------------------------------------+---------------------------+
| Prompt input       | Reads the requested scene.                     | No                        |
| Storyboard planner | Selects scene, narration, positions, motion.   | No, fixed Phase 1 rules.  |
| Asset generator    | Draws the images used in the scene.            | No, drawing code.         |
| Kokoro             | Turns narration text into a spoken voice.      | Yes, voice AI model.      |
| Rhubarb            | Finds speech sounds and times mouth shapes.    | Speech-analysis tool.     |
| Scene compiler     | Converts the plan into timed video layers.     | No                        |
| Validators and QA | Check duration, missing files, and timing.      | No                        |
| FFmpeg renderer    | Combines everything into an MP4 video.         | No, video tool.           |
+--------------------+------------------------------------------------+---------------------------+
```

## Phase 2 Architecture Update

The Phase 2 structural planner uses Groq's supported `openai/gpt-oss-20b` endpoint. It replaced the earlier Llama endpoint after Groq deprecated that endpoint for this account tier.

The project remains local-first: Whisper transcribes voice on this Mac; assets, artifact storage, previews, approval, and FFmpeg rendering stay on this Mac. Only plain-language planning text is sent to Groq. Groq returns a JSON animation blueprint, which is validated locally before any asset or rendering tool can use it. Audio files, images, video, file paths, FFmpeg commands, and API keys are never sent in planner requests.

## Current Phase 1 Limitation

The current planner is template-based, not a general AI video planner. A prompt containing `butterfly` and `flower` selects a fixed scene. That is why the first render placed both objects in the center, with the flower in front of the butterfly.

For a more natural scene, the planner and renderer should eventually support instructions such as:

- Place the butterfly on top of the flower.
- Make the butterfly wings flap around its body.
- Anchor the flower at the bottom of its stem.
- Make the flower head lean forward and backward in a gentle arc.

## Editable Bird and Fence Geometry (Phase 2)

New bird storyboards store validated `bird_fence_geometry` settings. Rejection feedback
can revise these settings; the pipeline regenerates the background and shared bird motion
before producing the next preview. Approval reuses the saved version's settings.

- Fence coordinates use a 1920 × 1080 canvas: `pole_x_positions`, `pole_top_y`,
  `pole_bottom_y`, `pole_width`, `rod_left_x`, `rod_right_x`, and `rod_thickness`.
- The rod's bottom equals `pole_top_y`; its top equals `pole_top_y - rod_thickness`.
  Pole and rod raster bounds are adjacent without overlapping pixels.
- `bird_scale`, `hop_start_x_percent`, `hop_end_x_percent`, `hop_height` (pixels),
  `hop_period_frames`, and `hop_count` control all bird parts together.
- Landing height is derived from the rod's surface and the visible foot boundary,
  including output scaling. Hop height measures upward from that landing surface.
  Hop sounds follow the same timing. Invalid or out-of-canvas geometry is rejected.

For example, request: "Set the pole tops to y=850, rod thickness to 80 pixels,
and hop height to 70 pixels; keep the other settings unchanged."
Restart the pipeline process to load code updates. Existing previews are immutable;
generate a new project or reject a saved preview to create a new version.
Saved storyboards predating the geometry field retain their original appearance on approval.
