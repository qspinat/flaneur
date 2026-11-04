// The 3D walk: today's street from Panoramax, the old photos on their facades, the years
// chosen on a slider. `#<lon>,<lat>,<heading>,<min>-<max>`.
import type { Maps } from '../map';
import { $, drawPlate, outLinks, typing, wirePlan } from '../page';
import { FIRST_YEAR, LAST_YEAR, format3d, hashOf, json, parse3d, yearsLabel, type Place3d } from '../walk';
import { Scene3D } from './scene';
import './walk3d.css';
import {
  inRange,
  nearby,
  nearestPanorama,
  nearestTo,
  photoToward,
  stepTarget,
  yearMarks,
  type OldPhoto,
  type Scene,
} from './walk3d';

const DRAG_PX = 5; // a press that moves less than this is a click
const URL_AFTER_MS = 300; // the link is written once the view rests
const REVIEW = 'http://127.0.0.1:8765/'; // `flaneur review`, on the user's computer

function link(href: string, text: string, newTab = false): HTMLAnchorElement {
  const a = Object.assign(document.createElement('a'), { href, textContent: text });
  if (newTab) Object.assign(a, { target: '_blank', rel: 'noopener' });
  return a;
}


/** What changed since the last frame: where one stands, the way one looks, the years. */
interface Changes {
  step: boolean;
  look: boolean;
  years: boolean;
}

class Walk3d {
  private view: Scene3D;
  private scene: Scene;
  private range: [number, number] = [FIRST_YEAR, LAST_YEAR];
  private showing = '';
  private maps: Maps | null = null;
  private stood = ''; // the panorama the page was last drawn for
  private pending: Changes = { step: false, look: false, years: false };
  private frame = 0;
  private urlTimer = 0;
  private planOpen: () => boolean;

  constructor(scene: Scene) {
    this.scene = scene;
    this.view = new Scene3D($('view'), scene);
    this.view.onChange = () => this.later({ step: this.view.here.id !== this.stood, look: true, years: false });
    for (const id of ['range-min', 'range-max']) Object.assign($<HTMLInputElement>(id), { min: FIRST_YEAR, max: LAST_YEAR });
    this.planOpen = wirePlan(
      () => this.openPlan(),
      () => this.closePlan(),
    );
    this.listen();
  }

  /** Where the link says, else the first panorama. */
  async start(): Promise<void> {
    const place = parse3d(location.hash);
    if (place) await this.goToPlace(place);
    else if (import.meta.env.DEV && location.hash.startsWith('#photo=')) await this.standAtPhoto();
    else await this.go(this.scene.panoramas[0].id, 0);
  }

  private async goToPlace({ lon, lat, heading, range }: Place3d): Promise<void> {
    this.setRange(range);
    this.later({ step: false, look: false, years: true }); // the same place may take new years
    await this.go(nearestTo(this.scene, lon, lat).id, heading);
  }

  /** In development, for the screenshots: `#photo=<id>` stands where that photo was taken,
   * `?aside=6` six metres to its left and two back. */
  private async standAtPhoto(): Promise<void> {
    const photo = this.scene.photos.find((p) => p.id === decodeURIComponent(location.hash.slice(7)));
    if (!photo) return this.go(this.scene.panoramas[0].id, 0);
    await this.go(nearestPanorama(this.scene, photo.pose.x, photo.pose.y).id, photo.pose.heading);
    const aside = new URLSearchParams(location.search).get('aside');
    if (aside === null) return;
    const r = (photo.pose.heading * Math.PI) / 180;
    const [left, back] = [Number(aside), 2];
    this.view.standAt(photo.pose.x - left * Math.cos(r) - back * Math.sin(r), photo.pose.y + left * Math.sin(r) - back * Math.cos(r), photo.pose.z);
  }

  private async go(id: string, heading?: number): Promise<void> {
    await this.view.goTo(id, heading);
    (window as unknown as { ready: boolean }).ready = true; // read by the screenshots
  }

  private listen(): void {
    const canvas = $('view');
    let drag: { x: number; y: number; heading: number; pitch: number; moved: number } | null = null;
    canvas.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, heading: this.view.heading, pitch: this.view.pitch, moved: 0 };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointercancel', () => (drag = null)); // the browser took the touch
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      drag.moved = Math.max(drag.moved, Math.hypot(e.clientX - drag.x, e.clientY - drag.y));
      const degreesPerPixel = 70 / canvas.clientHeight;
      this.view.look(drag.heading - (e.clientX - drag.x) * degreesPerPixel, drag.pitch + (e.clientY - drag.y) * degreesPerPixel);
    });
    canvas.addEventListener('pointerup', (e) => {
      const clicked = drag && Math.max(drag.moved, Math.hypot(e.clientX - drag.x, e.clientY - drag.y)) < DRAG_PX;
      drag = null;
      if (clicked) this.showCard(this.view.photoAt(e.clientX, e.clientY));
    });
    document.addEventListener('keydown', (e) => {
      if (this.planOpen() || typing(e)) return;
      if (e.key === 'ArrowLeft') this.view.look(this.view.heading - 15, this.view.pitch);
      else if (e.key === 'ArrowRight') this.view.look(this.view.heading + 15, this.view.pitch);
      else if (e.key === 'ArrowUp') {
        const next = stepTarget(this.view.here, this.view.heading);
        if (next) void this.go(next[0]);
      } else if (e.key === 'Escape') this.showCard(null);
      else if (e.key === 'm' || e.key === 'M') this.openPlan();
    });
    for (const id of ['range-min', 'range-max']) {
      $(id).addEventListener('input', () => {
        const low = Number($<HTMLInputElement>('range-min').value);
        const high = Number($<HTMLInputElement>('range-max').value);
        // the handle moved never passes the other one
        this.setRange(id === 'range-min' ? [Math.min(low, high), high] : [low, Math.max(low, high)]);
        this.later({ step: false, look: false, years: true });
      });
    }
    $('card-close').addEventListener('click', () => this.showCard(null));
    window.addEventListener('hashchange', () => {
      const place = parse3d(location.hash);
      if (place && location.hash !== this.hash()) void this.goToPlace(place);
    });
  }

  private setRange([min, max]: [number, number]): void {
    this.range = [Math.max(FIRST_YEAR, min), Math.min(LAST_YEAR, max)];
    $<HTMLInputElement>('range-min').value = String(this.range[0]);
    $<HTMLInputElement>('range-max').value = String(this.range[1]);
    const whole = this.range[0] === FIRST_YEAR && this.range[1] === LAST_YEAR;
    $('range-label').textContent = whole ? 'Toutes les années' : `${this.range[0]} – ${this.range[1]}`;
  }

  private hash(): string {
    const { lon, lat } = this.view.here;
    return format3d({ lon, lat, heading: this.view.heading, range: this.range });
  }

  /** Gather what changed and draw it once, at the next frame (a drag moves many times). */
  private later(changes: Changes): void {
    this.pending = {
      step: this.pending.step || changes.step,
      look: this.pending.look || changes.look,
      years: this.pending.years || changes.years,
    };
    if (!this.frame) this.frame = requestAnimationFrame(() => this.draw());
  }

  private draw(): void {
    this.frame = 0;
    if (!this.view.here) return; // not standing anywhere yet: kept for when the walk starts
    const { step, look, years } = this.pending;
    this.pending = { step: false, look: false, years: false };
    const { id, x, y, lon, lat } = this.view.here;
    if (step) {
      this.stood = id;
      this.buildWays();
      this.drawCredits(); // today's panorama is credited, whatever old photos are shown
      $('range-marks').replaceChildren(...yearMarks(this.scene.photos, x, y).map((year) => this.mark(year)));
    }
    if (step || years) void this.showPhotos();
    if (step || look) {
      this.placeWays();
      const toward = photoToward(this.scene.walk, x, y, this.view.heading);
      $<HTMLAnchorElement>('back').href = toward ? `./${hashOf(toward.id)}` : './';
      drawPlate(toward);
      this.maps?.show({ lon, lat, heading: this.view.heading });
    }
    window.clearTimeout(this.urlTimer);
    this.urlTimer = window.setTimeout(() => history.replaceState(null, '', this.hash()), URL_AFTER_MS);
  }

  private async showPhotos(): Promise<void> {
    const { x, y } = this.view.here;
    const seen = nearby(this.scene.photos.filter((p) => inRange(p, this.range)), x, y);
    const ids = seen.map(([p]) => p.id).join();
    if (ids !== this.showing) {
      this.showing = ids;
      await this.view.setPhotos(seen.map(([p]) => p));
      this.drawCredits();
    }
    const strength = new Map(seen.map(([p, s]) => [p.id, s]));
    this.view.setStrengths(this.view.photos.map((p) => strength.get(p.id) ?? 0));
  }

  private mark(year: number): HTMLElement {
    const mark = document.createElement('span');
    mark.style.left = `${((year - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)) * 100}%`;
    return mark;
  }

  /** The ways on from this panorama, once; placeWays puts them on the pavement. */
  private buildWays(): void {
    $('ways').replaceChildren(
      ...this.view.here.links.map(([id, bearing]) => {
        const b = Object.assign(document.createElement('button'), { type: 'button', className: 'way' });
        b.dataset.bearing = String(bearing);
        b.setAttribute('aria-label', 'Avancer par là');
        b.addEventListener('click', () => void this.go(id));
        return b;
      }),
    );
  }

  private placeWays(): void {
    for (const b of $('ways').children as HTMLCollectionOf<HTMLElement>) {
      const at = this.view.onScreen(Number(b.dataset.bearing), 4);
      b.hidden = !at;
      if (at) Object.assign(b.style, { left: `${at.x}px`, top: `${at.y}px` });
    }
  }

  private drawCredits(): void {
    const here = this.view.here;
    const today = document.createElement('p');
    today.append(`Rue d'aujourd'hui (${here.taken.slice(0, 4)}) : ${here.credit}. `, ...outLinks(here.url, 'Ouvrir dans Panoramax'));
    $('credits-list').replaceChildren(
      today,
      ...this.view.photos.map((photo) =>
        Object.assign(document.createElement('p'), {
          textContent: `${yearsLabel(photo.years) || 's. d.'}, ${photo.title}. ${photo.credit}.`,
        }),
      ),
    );
  }

  /** An old photo's card (null: closed): what it is, its credit, and where to see it. */
  private showCard(photo: OldPhoto | null): void {
    $('card').hidden = !photo;
    if (!photo) return;
    $('card-when').textContent = yearsLabel(photo.years) || 'Date inconnue';
    $('card-title').textContent = photo.title;
    $('card-credit').textContent = photo.credit;
    const links: (Node | string)[] = [link(`./${hashOf(photo.id)}`, 'Voir dans la balade')];
    // in the local version only: correct its place in the review page
    if (import.meta.env.DEV) links.push(' · ', link(`${REVIEW}#${encodeURIComponent(photo.id)}`, 'Corriger', true));
    $('card-links').replaceChildren(...links);
  }

  /** The maps, once the street is shown (they weigh more than the rest). */
  async loadMaps(maps: Promise<typeof import('../map')>): Promise<void> {
    const { Maps } = await maps;
    this.maps = new Maps($('minimap'), this.scene.walk);
    const { lon, lat } = this.view.here;
    this.maps.show({ lon, lat, heading: this.view.heading });
  }

  private openPlan(): void {
    if (!this.maps) return;
    $('plan').hidden = false;
    const { lon, lat } = this.view.here;
    this.maps.openPlan($('plan-map'), { lon, lat, heading: this.view.heading }, (id) => {
      this.closePlan();
      const photo = this.scene.walk.find((p) => p.id === id);
      if (photo) void this.go(nearestPanorama(this.scene, photo.x, photo.y).id, photo.heading ?? this.view.heading);
    });
    $('plan-close').focus();
  }

  private closePlan(): void {
    $('plan').hidden = true;
    $('minimap').focus();
  }
}

const maps = import('../map'); // fetched alongside the street, built after it
json<Scene>('scene.json')
  .then(async (scene) => {
    const walk3d = new Walk3d(scene);
    await walk3d.start();
    await walk3d.loadMaps(maps);
  })
  .catch(() => {
    document.body.append(
      Object.assign(document.createElement('p'), {
        id: 'failed',
        textContent: "La rue n'a pas pu être chargée. Rechargez la page dans un instant.",
      }),
    );
  });
