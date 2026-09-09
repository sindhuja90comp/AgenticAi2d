import React from 'react';
import { AbsoluteFill, Audio, Img, staticFile, useCurrentFrame } from 'remotion';

const activeClip = (clips, frame) => clips.find((clip) => frame >= clip.start_frame && frame < clip.start_frame + clip.duration_frames);

const motionStyle = (clip, frame) => {
  const motion = clip.motion;
  if (!motion || motion.kind === 'static') return {};
  const local = frame - clip.start_frame;
  if (motion.kind === 'butterfly_flap') {
    const flap = 1 - (motion.amplitude_percent / 100) * (0.5 + 0.5 * Math.sin((local / motion.period_frames) * Math.PI * 2));
    return { transform: `translate(-50%, -50%) scaleX(${flap})` };
  }
  if (motion.kind === 'bird_hop') {
    const progress = Math.min(1, Math.max(0, local / Math.max(1, clip.duration_frames - 1)));
    const hops = Math.sin(progress * Math.PI * motion.hop_count);
    const x = motion.start_x_percent + (motion.end_x_percent - motion.start_x_percent) * progress;
    return { left: `${x}%`, top: `${70 - Math.max(0, hops) * motion.amplitude_percent}%` };
  }
  if (motion.kind === 'flower_sway') {
    const angle = Math.sin((local / motion.period_frames) * Math.PI * 2) * motion.amplitude_percent;
    return { transform: `translate(-50%, -50%) rotate(${angle}deg)` };
  }
  return {};
};

export const TimelineComposition = ({ timeline, assets }) => {
  const frame = useCurrentFrame();
  if (!timeline) return null;
  const visualLayers = [...timeline.visual_tracks].sort((left, right) => left.z_index - right.z_index);
  const narration = timeline.audio_tracks[0];
  return (
    <AbsoluteFill style={{ backgroundColor: '#d8edf2', overflow: 'hidden' }}>
      {visualLayers.map((track) => {
        const clip = activeClip(track.clips, frame);
        if (!clip) return null;
          const style = {
          position: 'absolute',
          zIndex: track.z_index,
          left: `${clip.x_percent ?? 50}%`,
          top: `${clip.y_percent ?? 50}%`,
          transform: `translate(-50%, -50%) scale(${clip.scale ?? 1})`,
          width: track.kind === 'background' ? '100%' : '35%',
          height: track.kind === 'background' ? '100%' : 'auto',
            objectFit: 'cover',
            ...motionStyle(clip, frame),
        };
        if (track.kind === 'background') {
          style.left = '50%';
          style.top = '50%';
        }
        if (track.kind === 'character_mouth') {
          const viseme = assets.visemes[clip.viseme_timeline_id];
          const cue = viseme?.cues.find((item) => frame >= item.start_frame && frame <= item.end_frame);
          const shapeIndex = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X'].indexOf(cue?.mouth_shape ?? 'X');
          return (
            <div
              key={clip.clip_id}
              style={{
                ...style,
                width: '12%',
                height: '12%',
                backgroundImage: `url(${staticFile(assets.assets[clip.asset_id])})`,
                backgroundPosition: `0 ${(shapeIndex / 8) * 100}%`,
                backgroundRepeat: 'no-repeat',
                backgroundSize: '100% 900%',
              }}
            />
          );
        }
        return <Img key={clip.clip_id} src={staticFile(assets.assets[clip.asset_id])} style={style} />;
      })}
      {narration ? <Audio src={staticFile(assets.assets[narration.asset_id])} /> : null}
    </AbsoluteFill>
  );
};
