import type { Maps } from './map';
import { $, drawPlate, typing, wirePlan } from './page';
import {
  DATA,
  FIRST_YEAR,
  LAST_YEAR,
  distanceLabel,
  erasOf,
  format3d,
  hashOf,
  json,
  largest,
  photoIdFrom,
  span,
  srcsetOf,
  startOf,
  yearsLabel,
  type Direction,
  type Photo,
  type Walk,
} from './walk';

const AROUND_YEARS = 5; // the 3D walk opens on the photo's years, and this many either side

/** The 3D walk at a photo's place, looking its way, around its years. */
function in3d(photo: Photo): string {
  const years = span(photo.years);
  const range: [number, number] = years ? [years[0] - AROUND_YEARS, years[1] + AROUND_YEARS] : [FIRST_YEAR, LAST_YEAR];
  return format3d({ lon: photo.lon, lat: photo.lat, heading: photo.heading ?? 0, range });
}

const WAYS: { way: Direction; label: string; key: string; arrow: string }[] = [
  // chevrons drawn as on Street View's pavement: up, down, left, right
  { way: 'forward', label: 'Avancer', key: 'ArrowUp', arrow: 'M8 34 L27 16 L46 34' },
  { way: 'back', label: 'Reculer', key: 'ArrowDown', arrow: 'M8 20 L27 38 L46 20' },
  { way: 'left', label: 'Tourner à gauche', key: 'ArrowLeft', arrow: 'M34 8 L16 27 L34 46' },
  { way: 'right', label: 'Tourner à droite', key: 'ArrowRight', arrow: 'M20 8 L38 27 L20 46' },
];

// how wide the photograph is shown (see style.css): all the width on a phone
const SIZES = '(max-width: 820px) 100vw, 75vw';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function button(className: string, text: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  b.textContent = text;
  b.addEventListener('click', onClick);
  return b;
}

function outLink(href: string, text: string, className = ''): HTMLAnchorElement {
  return Object.assign(document.createElement('a'), { href, textContent: text, className, target: '_blank', rel: 'noopener' });
}

class Walker {
  private byId: Map<string, Photo>;
  private photos: Photo[];
  private here!: Photo;
  private showingToday = false;
  private maps: Maps | null = null;
  private moving = false;
  private planOpen: () => boolean;

  constructor(photos: Photo[], start: Photo) {
    this.photos = photos;
    this.byId = new Map(photos.map((p) => [p.id, p]));
    window.addEventListener('hashchange', () => {
      const id = photoIdFrom(location.hash);
      const photo = id ? this.byId.get(id) : undefined;
      if (photo && photo !== this.here) this.arrive(photo);
    });
    this.planOpen = wirePlan(
      () => this.openPlan(),
      () => this.closePlan(),
    );
    document.addEventListener('keydown', (event) => this.onKey(event));
    // the first photograph alone, then the maps (most of the code's weight)
    $<HTMLImageElement>('photo').addEventListener(
      'load',
      () => {
        performance.mark('first-photo'); // measured by tests/visual/walk.mjs
        void import('./map').then(({ Maps }) => {
          this.maps = new Maps($('minimap'), this.photos);
          this.maps.show(this.here);
        });
      },
      { once: true },
    );
    this.arrive(start);
  }

  private onKey(event: KeyboardEvent): void {
    if (this.planOpen() || typing(event)) return;
    const way = WAYS.find((w) => w.key === event.key);
    if (way) {
      event.preventDefault();
      this.step(way.way);
    } else if (event.key === 't' || event.key === 'T') this.toggleToday();
    else if (event.key === 'm' || event.key === 'M') this.openPlan();
  }

  /** Walk one step: the picture moves the way I go, then the next one comes in. */
  step(way: Direction): void {
    const link = this.here.links[way];
    const next = link && this.byId.get(link.id);
    if (!next || this.moving) return;
    if (reducedMotion.matches) {
      this.go(next);
      return;
    }
    this.moving = true;
    const stage = $('stage');
    const photo = $('photo');
    let fallback = 0;
    const arrived = () => {
      window.clearTimeout(fallback);
      photo.removeEventListener('transitionend', arrived);
      stage.classList.remove(`going-${way}`);
      this.moving = false;
      this.go(next);
    };
    photo.addEventListener('transitionend', arrived);
    fallback = window.setTimeout(arrived, 600); // in case no transition ends
    stage.classList.add(`going-${way}`);
  }

  private go(photo: Photo): void {
    history.pushState(null, '', hashOf(photo.id));
    this.arrive(photo);
  }

  private arrive(photo: Photo): void {
    this.here = photo;
    this.showingToday = false;
    history.replaceState(null, '', hashOf(photo.id));
    const img = $<HTMLImageElement>('photo');
    img.classList.add('loading');
    img.onload = () => {
      img.classList.remove('loading');
      // the photographs one step away, once this one is shown
      for (const link of Object.values(photo.links)) {
        const p = link && this.byId.get(link.id);
        if (p) Object.assign(new Image(), { fetchPriority: 'low', sizes: SIZES, srcset: srcsetOf(p.images) });
      }
    };
    drawPlate(photo);
    this.drawSteps(photo);
    this.drawAbout();
    this.maps?.show(photo);
  }

  private drawSteps(photo: Photo): void {
    $('steps').replaceChildren(
      ...WAYS.filter(({ way }) => photo.links[way]).map(({ way, label, arrow }) => {
        const link = photo.links[way]!;
        const step = button('step', '', () => this.step(way));
        step.dataset.way = way;
        step.setAttribute('aria-label', `${label} (${distanceLabel(link.distance_m)})`);
        step.innerHTML =
          `<svg viewBox="0 0 54 54" aria-hidden="true"><path d="${arrow}" fill="none" ` +
          'stroke="currentColor" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        if (way === 'forward' || way === 'back') {
          step.append(Object.assign(document.createElement('span'), { textContent: distanceLabel(link.distance_m) }));
        }
        return step;
      }),
    );
  }

  /** The picture and what is said of it: the archive photograph, or the view of today. */
  private drawAbout(): void {
    const photo = this.here;
    const today = this.showingToday ? photo.today : null;
    const img = $<HTMLImageElement>('photo');
    img.sizes = SIZES;
    img.srcset = today ? '' : srcsetOf(photo.images);
    img.src = DATA + (today ? today.image : largest(photo.images));
    img.alt = today ? "Le même endroit aujourd'hui" : photo.title;
    $('when').textContent = today ? today.taken.slice(0, 4) : yearsLabel(photo.years) || 'Date inconnue';
    $('title').textContent = photo.title;
    $('caption').textContent = photo.caption ?? '';
    $('credit').replaceChildren(
      ...(today
        ? [`Vue d'aujourd'hui : ${today.credit}. `, outLink(today.url, 'Ouvrir dans Panoramax')]
        : [
            `${photo.photographer}. ${photo.credit}. `,
            ...(photo.record_url ? [outLink(photo.record_url, 'Voir la notice du musée')] : []),
          ]),
    );
    this.drawEras(photo);
  }

  private drawEras(photo: Photo): void {
    const eras = erasOf(photo, this.byId);
    const buttons: HTMLElement[] =
      eras.length > 1
        ? eras.map((p) => {
            const era = button('era', yearsLabel(p.years) || 's. d.', () => p !== photo && this.go(p));
            era.setAttribute('aria-pressed', String(p === photo && !this.showingToday));
            return era;
          })
        : [];
    buttons.push(Object.assign(document.createElement('a'), { href: `/3d.html${in3d(photo)}`, textContent: 'En 3D', className: 'era three' }));
    if (photo.today) {
      const today = button('era today', "Aujourd'hui", () => this.toggleToday());
      today.setAttribute('aria-pressed', String(this.showingToday));
      buttons.push(today);
    } else {
      // no view of today matched for sure: Panoramax, open at this spot
      const map = `https://api.panoramax.xyz/#focus=map&map=19/${photo.lat}/${photo.lon}`;
      buttons.push(outLink(map, "Aujourd'hui, sur Panoramax", 'era today'));
    }
    $('eras').replaceChildren(...buttons);
  }

  /** Swap the photograph for the view of the same place today, in the same frame. */
  toggleToday(): void {
    if (!this.here.today) return;
    this.showingToday = !this.showingToday;
    this.drawAbout();
  }

  private openPlan(): void {
    if (!this.maps) return;
    $('plan').hidden = false;
    this.maps.openPlan($('plan-map'), this.here, (id) => {
      this.closePlan();
      const photo = this.byId.get(id);
      if (photo) this.go(photo);
    });
    $('plan-close').focus();
  }

  private closePlan(): void {
    $('plan').hidden = true;
    $('minimap').focus();
  }
}

json<Walk>('walk.json')
  .then((walk) => new Walker(walk.photos, startOf(walk, location.hash)))
  .catch(() => {
    $('title').textContent =
      "Les photographies n'ont pas pu être chargées. Rechargez la page dans un instant.";
  });
