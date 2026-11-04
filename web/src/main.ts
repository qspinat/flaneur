import type { Maps, Plan } from './map';
import { $, Years, drawPlate, outLinks, typing } from './page';
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
  shownIn,
  yearsTaken,
  whenLabel,
  yearsLabel,
  type Direction,
  type Photo,
  type Walk,
} from './walk';

const AROUND_YEARS = 5; // the 3D walk opens on the photo's years, and this many either side
const PLAN_LAST = 1950; // the plan's years end here, or at the last year a photograph has

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

/** What lies over the plan, to keep clear around what it shows: what spans the width or is low
 *  (the plate, the years, a phone's sheets) clears the top or the bottom; a tall panel (the
 *  welcome, the print, on a laptop) clears its side. */
function overlaysPadding(): { top: number; bottom: number; left: number; right: number } {
  const gap = 16;
  const [w, h] = [window.innerWidth, window.innerHeight];
  const pad = { top: gap, bottom: gap, left: gap, right: gap };
  for (const id of OVERLAYS) {
    if ($(id).hidden) continue;
    const r = $(id).getBoundingClientRect();
    if (isSheet(r) || r.height < h * 0.2) {
      if (r.top + r.height / 2 < h / 2) pad.top = Math.max(pad.top, r.bottom + gap);
      else pad.bottom = Math.max(pad.bottom, h - r.top + gap);
    } else if (r.left + r.width / 2 < w / 2) pad.left = Math.max(pad.left, r.right + gap);
    else pad.right = Math.max(pad.right, w - r.left + gap);
  }
  return pad;
}

const OVERLAYS = ['brand', 'plan-years', 'welcome', 'print']; // what lies over the plan

/** An overlay spanning the plan's width (a phone's sheet, at the bottom; the years, at the top). */
function isSheet(r: DOMRect): boolean {
  return r.width > window.innerWidth * 0.6;
}

/** The plan's credit (its licence asks for it) kept above any sheet along the bottom: the
 *  sheets are watched, so that no change to them has to remember it. */
function keepCreditClear(): void {
  const place = () => {
    const h = window.innerHeight;
    const low = OVERLAYS.map((id) => $(id).getBoundingClientRect()).filter((r) => isSheet(r) && r.top > h / 2);
    $('home').style.setProperty('--sheet', `${h - Math.min(h, ...low.map((r) => r.top))}px`);
  };
  const watch = new ResizeObserver(place); // a hidden sheet shrinks to nothing: seen too
  for (const id of ['welcome', 'print']) watch.observe($(id));
  window.addEventListener('resize', place); // a sheet the same size, moved by the layout
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
  private years: Years; // which photographs the plan shows
  private yearsFrame = 0;
  private refocus: string | null = null; // the way's selector to focus on the next photo (keyboard)

  constructor(photos: Photo[]) {
    this.photos = photos;
    this.byId = new Map(photos.map((p) => [p.id, p]));
    keepCreditClear();
    const last = Math.max(PLAN_LAST, ...yearsTaken(photos).slice(-1));
    this.years = new Years($('plan-years'), FIRST_YEAR, last, () => this.showYears());
    // a mark at the year each photograph was taken (where it spans decades, its first)
    this.years.mark([...new Set(photos.flatMap((p) => span(p.years)?.[0] ?? []))]);
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
    const wayOf = (el: EventTarget | null) => (el as HTMLElement | null)?.closest?.<HTMLElement>('.step, .onward')?.dataset.way;
    for (const type of ['pointerover', 'focusin', 'pointerout', 'focusout'] as const) {
      $('walk').addEventListener(type, (event) => {
        const way = wayOf(event.target);
        if (!way || wayOf(event.relatedTarget) === way) return; // within the same way: nothing changes
        const on = type === 'pointerover' || type === 'focusin';
        for (const el of $('walk').querySelectorAll<HTMLElement>(`[data-way="${way}"]`)) el.classList.toggle('lit', on);
      });
    }
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
      this.showYears();
    }
    this.plan.shown();
    if (photo && shownIn(photo.years, this.years.range, this.years.whole)) this.choose(photo, true);
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

  /** The plan shows only the photographs of these years (the one chosen, if it is not, set
   *  aside); while a handle is dragged, once a frame at most. */
  private showYears(): void {
    if (this.yearsFrame) return;
    this.yearsFrame = requestAnimationFrame(() => {
      this.yearsFrame = 0;
      const [range, whole] = [this.years.range, this.years.whole];
      this.plan?.only(whole ? null : this.photos.filter((p) => shownIn(p.years, range, whole)).map((p) => p.id));
      if (this.chosen && !shownIn(this.chosen.years, range, whole)) this.choose(null);
    });
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
    const fromPlan = !$('home').hidden; // (hidden until the address is read: see index.html)
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
    // a way taken from the keyboard: the same way stays in focus on the next photo
    const from = document.activeElement?.closest('.step, .onward');
    this.refocus = from ? `.${from.classList.contains('step') ? 'step' : 'onward'}[data-way="${way}"]` : null;
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
    this.drawWays(photo);
    if (this.refocus) {
      ($('walk').querySelector<HTMLElement>(this.refocus) ?? $('to-plan')).focus({ preventScroll: true });
      this.refocus = null;
    }
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

  /** Where each way leads: the photograph there, small, with how far and when; laid out as
   *  the arrows are. Pointing at an arrow lights its way, and the way its arrow. */
  private drawWays(photo: Photo): void {
    $('onwards').replaceChildren(
      ...WAYS.flatMap(({ way, label }) => {
        const link = photo.links[way];
        const there = link && this.byId.get(link.id);
        if (!there) return [];
        const item = button('onward', '', () => this.step(way));
        item.dataset.way = way;
        const img = Object.assign(document.createElement('img'), {
          alt: '',
          loading: 'lazy',
          sizes: '112px',
          srcset: srcsetOf(there.images),
          src: DATA + largest(there.images),
        });
        const words = document.createElement('span');
        words.className = 'onward-words';
        words.append(
          Object.assign(document.createElement('span'), { className: 'onward-label', textContent: label }),
          Object.assign(document.createElement('span'), {
            className: 'onward-when',
            textContent: `${distanceLabel(link.distance_m)}, ${yearsLabel(there.years) || 'date inconnue'}`,
          }),
        );
        item.append(img, words);
        item.setAttribute('aria-label', `${label} : ${there.title}, ${whenLabel(there.years)}, à ${distanceLabel(link.distance_m)}`);
        return item;
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
    $('home').hidden = false;
    $('welcome').hidden = false;
    $('welcome-text').textContent = "Les photographies n'ont pas pu être chargées. Rechargez la page dans un instant.";
  });
