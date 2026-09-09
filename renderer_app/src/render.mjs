import { bundle } from '@remotion/bundler';
import { renderMedia, selectComposition } from '@remotion/renderer';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const [timelinePath, assetMapPath, outputPath] = process.argv.slice(2);
if (!timelinePath || !assetMapPath || !outputPath) {
  throw new Error('Usage: node render.mjs <timeline.json> <asset-map.json> <output.mp4>');
}

const timeline = JSON.parse(await fs.readFile(timelinePath, 'utf8'));
const assets = JSON.parse(await fs.readFile(assetMapPath, 'utf8'));
const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
const serveUrl = await bundle({ entryPoint: path.join(currentDirectory, 'index.jsx') });
const inputProps = { timeline, assets };
const selected = await selectComposition({
  serveUrl,
  id: 'TimelineComposition',
  inputProps,
});
const composition = {
  ...selected,
  width: timeline.width,
  height: timeline.height,
  fps: timeline.fps,
  durationInFrames: timeline.duration_frames,
};
await renderMedia({
  serveUrl,
  composition,
  codec: 'h264',
  outputLocation: outputPath,
  inputProps,
});
