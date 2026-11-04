import type { Maps, Plan } from './map';
import { $, drawPlate, outLinks, typing } from './page';
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
  richest,
  whenLabel,
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


/** The plan's welcome is shown until the visitor sets it aside (remembered by this browser). */
const WELCOMED = 'flaneur.welcomed';
function welcomed(): boolean {
  try {
    return localStorage.getItem(WELCOMED) === '1';
  } catch {
    return false; // storage refused (private window…): welcome again
  }
}
function rememberWelcomed(): void {
  try {
    localStorage.setItem(WELCOMED, '1');
  } catch {
    // nothing to remember with
  }
}

/** A change of view that the browser animates when it can (the print growing into the
 *  photograph): the transition's name is given to `from` before the change, to `to` after. */
function morph(from: HTMLElement, to: HTMLElement, change: () => void): void {
  if (!document.startViewTransition || reducedMotion.matches) {
    change();
    return;
  }
  from.style.viewTransitionName = 'print';
  const transition = document.startViewTransition(() => {
    from.style.viewTransitionName = '';
    to.style.viewTransitionName = 'print';
    change();
  });
  void transition.finished.finally(() => (to.style.viewTransitionName = ''));
}

/** What lies over the plan, to keep clear around what it shows: the plate along the top, and
 *  the welcome or the print, a sheet along the bottom (a phone) or a panel at one side. */
function overlaysPadding(): { top: number; bottom: number; left: number; right: number } {
  const gap = 16;
  const pad = { top: $('brand').getBoundingClientRect().bottom + gap, bottom: gap, left: gap, right: gap };
  for (const id of ['welcome', 'print']) {
    if ($(id).hidden) continue;
    const r = $(id).getBoundingClientRect();
    if (r.width > window.innerWidth * 0.8) pad.bottom = Math.max(pad.bottom, window.innerHeight - r.top + gap);
    else if (r.left < window.innerWidth / 2) pad.left = Math.max(pad.left, r.right + gap);
    else pad.right = Math.max(pad.right, window.innerWidth - r.left + gap);
  }
  return pad;
}

// the home page is the plan: its code is fetched at once, beside the photographs' list
const planCode = photoIdFrom(location.hash) ? undefined : import('./map');

class Walker {
  private byId: Map<string, Photo>;
  private photos: Photo[];
  private here!: Photo;
  private showingToday = false;
  private maps: Maps | null = null;
  private plan: Plan | null = null;
  private chosen: Photo | null = null; // the photograph shown on the plan, if any
  private moving = false;

  constructor(photos: Photo[]) {
    this.photos = photos;
    this.byId = new Map(photos.map((p) => [p.id, p]));
    // the address says what is shown: the plan (#/), or one photograph (#/<id>)
    window.addEventListener('hashchange', () => this.route());
    window.addEventListener('resize', () => !$('home').hidden && this.plan?.shown());
    for (const id of ['to-plan', 'plate', 'brand']) $(id).addEventListener('click', (e) => this.toPlan(e));
    $('minimap').addEventListener('click', () => this.toPlan());
    $('minimap').addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && this.toPlan());
    $('start').addEventListener('click', () => {
      this.setWelcomeAside();
      this.enter(richest(this.photos));
    });
    $('welcome-close').addEventListener('click', () => this.setWelcomeAside());
    $('print-close').addEventListener('click', () => this.choose(null));
    for (const id of ['enter', 'print-open']) $(id).addEventListener('click', () => this.chosen && this.enter(this.chosen));
    const about = $<HTMLDialogElement>('about-site');
    $('about-open').addEventListener('click', () => about.showModal());
    $('about-close').addEventListener('click', () => about.close());
    about.addEventListener('click', (e) => e.target === about && about.close()); // outside its box
    document.addEventListener('keydown', (event) => this.onKey(event));
    // a photograph first: it alone, then the maps (most of the code's weight)
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
    this.route();
  }

  /** Show what the address names: a photograph, or else the plan (with the last photograph
   *  seen chosen on it, however one came back to it). */
  private route(): void {
    const id = photoIdFrom(location.hash);
    const photo = id ? this.byId.get(id) : undefined;
    if (photo) this.showPhoto(photo);
    else void this.showPlan(this.here as Photo | undefined);
  }

  private onKey(event: KeyboardEvent): void {
    if (typing(event) || $<HTMLDialogElement>('about-site').open) return;
    if (!$('home').hidden) {
      if (event.key === 'Escape') this.choose(null);
      return;
    }
    const way = WAYS.find((w) => w.key === event.key);
    if (way) {
      event.preventDefault();
      this.step(way.way);
    } else if (event.key === 't' || event.key === 'T') this.toggleToday();
    else if (event.key === 'm' || event.key === 'M' || event.key === 'Escape') this.toPlan();
  }

  /** From a photograph back to the plan, on it (on the plan already: nothing to do). */
  private toPlan(event?: Event): void {
    event?.preventDefault();
    if (!$('home').hidden) return;
    history.pushState(null, '', '#/');
    this.route();
  }

  /** The plan, with `photo` chosen on it (none: the welcome, until set aside). */
  private async showPlan(photo?: Photo): Promise<void> {
    $('walk').hidden = true;
    $('home').hidden = false;
    $('welcome').hidden = !!photo || !!this.chosen || welcomed();
    if (!this.plan) {
      const { Plan } = await (planCode ?? import('./map'));
      if ($('home').hidden) return; // the address changed meanwhile: the plan waits to be seen
      const choose = (id: string | null) => this.choose(id ? (this.byId.get(id) ?? null) : null);
      this.plan ??= new Plan($('map'), this.photos, choose, overlaysPadding());
    }
    this.plan.shown();
    if (photo) this.choose(photo, true);
    ($('print').hidden ? $('brand') : $('enter')).focus({ preventScroll: true });
  }

  /** A photograph chosen on the plan: its print, laid over the plan. */
  private choose(photo: Photo | null, reveal = false): void {
    this.chosen = photo;
    $('print').hidden = !photo;
    if (photo) $('welcome').hidden = true;
    this.plan?.select(photo);
    if (!photo) return;
    // the photograph itself, fetched meanwhile: entering it finds it ready
    Object.assign(new Image(), { fetchPriority: 'low', sizes: SIZES, srcset: srcsetOf(photo.images) });
    const img = $<HTMLImageElement>('print-img');
    img.srcset = srcsetOf(photo.images);
    img.src = DATA + largest(photo.images);
    img.alt = photo.title;
    $('print-when').textContent = whenLabel(photo.years);
    $('print-title').textContent = photo.title;
    if (reveal) {
      // once the print has its picture (and the sheet its height): what is left clear is known
      const place = () => this.chosen === photo && this.plan?.reveal(photo, overlaysPadding());
      void img.decode().then(place, place);
    }
  }

  /** Into a photograph from the plan: its print grows into it. */
  private enter(photo: Photo): void {
    history.pushState(null, '', hashOf(photo.id));
    morph($('print-img'), $('photo'), () => this.showPhoto(photo));
  }

  private setWelcomeAside(): void {
    rememberWelcomed();
    $('welcome').hidden = true;
  }

  private showPhoto(photo: Photo): void {
    const fromPlan = !$('home').hidden;
    $('home').hidden = true;
    $('walk').hidden = false;
    if (photo !== this.here) this.arrive(photo);
    if (fromPlan) $('to-plan').focus({ preventScroll: true }); // focus leaves the hidden plan
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
    $('when').textContent = today ? today.taken.slice(0, 4) : whenLabel(photo.years);
    $('title').textContent = photo.title;
    $('caption').textContent = photo.caption ?? '';
    $('credit').replaceChildren(
      ...(today
        ? [`Vue d'aujourd'hui : ${today.credit}. `, ...outLinks(today.url, 'Ouvrir dans Panoramax')]
        : [`${photo.photographer}. ${photo.credit}. `, ...outLinks(photo.record_url, 'Voir la notice du musée')]),
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
    if (photo.in_3d) {
      buttons.push(Object.assign(document.createElement('a'), { href: `3d.html${in3d(photo)}`, textContent: 'En 3D', className: 'era three' }));
    }
    if (photo.today) {
      const today = button('era today', "Aujourd'hui", () => this.toggleToday());
      today.setAttribute('aria-pressed', String(this.showingToday));
      buttons.push(today);
    } else {
      // no view of today matched for sure: Panoramax, open at this spot
      const map = `https://api.panoramax.xyz/#focus=map&map=19/${photo.lat}/${photo.lon}`;
      buttons.push(...outLinks(map, "Aujourd'hui, sur Panoramax", 'era today'));
    }
    $('eras').replaceChildren(...buttons);
  }

  /** Swap the photograph for the view of the same place today, in the same frame. */
  toggleToday(): void {
    if (!this.here.today) return;
    this.showingToday = !this.showingToday;
    this.drawAbout();
  }
}

json<Walk>('walk.json')
  .then((walk) => new Walker(walk.photos))
  .catch(() => {
    $('welcome').hidden = false;
    $('welcome-text').textContent = "Les photographies n'ont pas pu être chargées. Rechargez la page dans un instant.";
  });
