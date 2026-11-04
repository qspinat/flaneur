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
