// An Albert-Kahn plate as scanned: what of the scan is the plate (its frame and handwritten
// number left out) and where the museum's label lies on it, as shares of the image (x0, y0, x1,
// y1, from the top left; see the pipeline's plates.py). Other photos have neither.
export type Box = [number, number, number, number];

export interface Framed {
  plate?: Box;
  label?: Box;
}

const WHOLE: Box = [0, 0, 1, 1];
const NOTHING: Box = [-1, -1, -1, -1]; // an empty box: no pixel of the photo falls in it

/** What the 3D walk draws of a photo: its plate (else the whole of it), its label never. */
export function framing(photo: Framed): { plate: Box; label: Box } {
  return { plate: photo.plate ?? WHOLE, label: photo.label ?? NOTHING };
}

const inside = ([x0, y0, x1, y1]: Box, u: number, v: number) => u >= x0 && u <= x1 && v >= y0 && v <= y1;

/** Whether the point (u, v) of a photo (shares of its image) is drawn: as the 3D walk's
 *  shaders (photo.ts, PLATE) decide it. */
export function drawn(photo: Framed, u: number, v: number): boolean {
  const { plate, label } = framing(photo);
  return inside(plate, u, v) && !inside(label, u, v);
}

/** The photo walk's image cut to its plate (CSS clip-path; none: shown whole), in pixels of
 *  the element: the image fitted in it whole (object-fit: contain), centred. The label stays:
 *  the museum's credit, on the photo where the museum put it. */
export function clipPath(plate: Box | undefined, box: { width: number; height: number }, image: { width: number; height: number }): string {
  if (!plate || !image.width || !image.height) return '';
  const scale = Math.min(box.width / image.width, box.height / image.height);
  const [w, h] = [image.width * scale, image.height * scale];
  const [left, top] = [(box.width - w) / 2, (box.height - h) / 2];
  const [x0, y0, x1, y1] = plate;
  const px = (v: number) => `${Math.max(0, Math.ceil(v))}px`; // outward: no line of frame left
  return `inset(${px(top + y0 * h)} ${px(box.width - left - x1 * w)} ${px(box.height - top - y1 * h)} ${px(left + x0 * w)})`;
}

interface Cut {
  plate?: Box;
  aspect?: number | null; // the image's height over width, known before it loads (else read once loaded)
}

const cuts = new WeakMap<HTMLImageElement, Cut>();
let resized: ResizeObserver | undefined;

/** Cut an image of the photo walk to a plate (none: shown whole), now and whenever its element
 *  changes size (or, its aspect unknown, once it has loaded). */
export function clipToPlate(img: HTMLImageElement, cut: Cut): void {
  if (!cuts.has(img)) {
    resized ??= new ResizeObserver((entries) => entries.forEach((e) => clip(e.target as HTMLImageElement)));
    resized.observe(img);
    img.addEventListener('load', () => clip(img));
  }
  cuts.set(img, cut);
  clip(img);
}

function clip(img: HTMLImageElement): void {
  const { plate, aspect } = cuts.get(img) ?? {};
  const image = aspect ? { width: 1, height: aspect } : { width: img.naturalWidth, height: img.naturalHeight };
  img.style.clipPath = clipPath(plate, { width: img.clientWidth, height: img.clientHeight }, image);
}
