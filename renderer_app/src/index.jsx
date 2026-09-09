import { Composition, registerRoot } from 'remotion';
import { TimelineComposition } from './timeline-composition.jsx';

export const RemotionRoot = () => {
  return (
    <Composition
      id="TimelineComposition"
      component={TimelineComposition}
      durationInFrames={1}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{ timeline: null, assets: {} }}
    />
  );
};

registerRoot(RemotionRoot);
