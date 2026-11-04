// What both walks share on screen: the fonts (served by the site itself: a third-party
// stylesheet would hold the first picture back by a connection to another host, measured 1 s
// on a slow 4G), the enamel plate, and the plan behind the minimap.
import '@fontsource/barlow-condensed/latin-500.css';
import '@fontsource/barlow-condensed/latin-600.css';
import '@fontsource/spectral/latin-400.css';
import '@fontsource/spectral/latin-400-italic.css';
import '@fontsource/spectral/latin-500.css';
import './style.css';
import { arrondissementLabel, httpsUrl } from './walk';

export const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** The street's enamel plate (hidden when the street is not known). */
export function drawPlate(place: { street: [string, string] | null; arrondissement: number | null } | null): void {
  const plate = $<HTMLAnchorElement>('plate');
  plate.hidden = !place?.street;
  if (!place?.street) return;
  const [kind, name] = place.street;
  $('plate-arrt').textContent = arrondissementLabel(place.arrondissement);
  $('plate-kind').textContent = kind;
  $('plate-name').textContent = name;
  plate.setAttribute('aria-label', `${kind} ${name}, voir le plan`);
}

/** The plan opens from the minimap (click, Enter or Space) and the plate, and closes with its
 * button or Escape; `open` and `close` do the rest. Returns whether the plan is open. */
export function wirePlan(open: () => void, close: () => void): () => boolean {
  const isOpen = () => !$('plan').hidden;
  $('minimap').addEventListener('click', open);
  $('minimap').addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && open());
  $('plate').addEventListener('click', (event) => {
    event.preventDefault();
    open();
  });
  $('plan-close').addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (!isOpen() || e.key !== 'Escape') return;
    close();
    e.stopImmediatePropagation(); // the walk's own keys do not see it
  });
  return isOpen;
}

/** Keys typed into a field are not the walk's. */
export function typing(event: KeyboardEvent): boolean {
  return !!(event.target as HTMLElement).closest('summary, input, textarea');
}

/** A link out of the site, in a new tab: none when the address is not https (one from the
 *  data that would be `javascript:` would run on this page). */
export function outLinks(href: string | null, text: string, className = ''): HTMLAnchorElement[] {
  const safe = httpsUrl(href);
  if (!safe) return [];
  return [Object.assign(document.createElement('a'), { href: safe, textContent: text, className, target: '_blank', rel: 'noopener' })];
}

/** Two handles on one bar to choose the years shown, from `first` to `last`, with a mark at each
 *  year a photograph was taken (see `mark`); `changed` is told of each choice. It is built in
 *  `root` (an empty form, class "years", named by its aria-label). */
export class Years {
  range: [number, number];
  private first: number;
  private last: number;
  private label = Object.assign(document.createElement('output'), { className: 'years-label' });
  private marks = Object.assign(document.createElement('div'), { className: 'years-marks' });
  private min = Object.assign(document.createElement('input'), { type: 'range', className: 'years-min' });
  private max = Object.assign(document.createElement('input'), { type: 'range', className: 'years-max' });

  constructor(root: HTMLElement, first: number, last: number, changed: (range: [number, number]) => void) {
    this.first = first;
    this.last = last;
    this.marks.setAttribute('aria-hidden', 'true');
    this.min.setAttribute('aria-label', 'À partir de');
    this.max.setAttribute('aria-label', "Jusqu'à");
    const track = Object.assign(document.createElement('div'), { className: 'years-track' });
    track.append(this.marks, this.min, this.max);
    root.replaceChildren(this.label, track);
    this.range = [first, last];
    for (const input of [this.min, this.max]) {
      Object.assign(input, { min: first, max: last });
      input.addEventListener('input', () => {
        const low = Number(this.min.value);
        const high = Number(this.max.value);
        // the handle moved never passes the other one
        this.set(input === this.min ? [Math.min(low, high), high] : [low, Math.max(low, high)]);
        changed(this.range);
      });
    }
    this.set(this.range);
  }

  /** Every year chosen. */
  get whole(): boolean {
    return this.range[0] === this.first && this.range[1] === this.last;
  }

  set([min, max]: [number, number]): void {
    this.range = [Math.max(this.first, min), Math.min(this.last, max)];
    this.min.value = String(this.range[0]);
    this.max.value = String(this.range[1]);
    this.label.textContent = this.whole ? 'Toutes les années' : `${this.range[0]} – ${this.range[1]}`;
  }

  /** A mark on the bar at each of these years. */
  mark(years: number[]): void {
    this.marks.replaceChildren(
      ...years.map((year) => {
        const mark = document.createElement('span');
        mark.style.left = `${((year - this.first) / (this.last - this.first)) * 100}%`;
        return mark;
      }),
    );
  }
}

/** A button of this class, saying `text`, doing `onClick`. */
export function button(className: string, text: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  b.textContent = text;
  b.addEventListener('click', onClick);
  return b;
}
