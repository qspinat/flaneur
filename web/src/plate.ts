// An Albert-Kahn plate as scanned: what of the scan is the plate (its frame and handwritten
// number left out) and where the museum's label lies on it, as shares of the image (x0, y0, x1,
// y1, from the top left; see the pipeline's plates.py). Other photos have neither.
export type Box = [number, number, number, number];

export interface Framed {
  plate?: Box | null;
  label?: Box | null;
}

const WHOLE: Box = [0, 0, 1, 1];
const NOTHING: Box = [-1, -1, -1, -1]; // an empty box: no pixel of the photo falls in it

/** What the 3D walk draws of a photo: its plate (else the whole of it), its label never. */
export function framing(photo: Framed): { plate: Box; label: Box } {
  return { plate: photo.plate ?? WHOLE, label: photo.label ?? NOTHING };
}

/** The photo walk's image cut to its plate (CSS clip-path; none: shown whole). The label stays:
 *  the museum's credit, on the photo where the museum put it. */
export function clipPath(plate: Box | null | undefined): string {
  if (!plate) return '';
  const [x0, y0, x1, y1] = plate;
  const percent = (v: number) => `${Math.round(v * 1000) / 10}%`;
  return `inset(${percent(y0)} ${percent(1 - x1)} ${percent(1 - y1)} ${percent(x0)})`;
}
