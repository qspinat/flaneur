// The 3D walk: today's panoramas from Panoramax, walked one to the next with a fade, the old
// photos placed in them drawn where they line up, the years chosen on a slider.
// `#<lon>,<lat>,<heading>,<min>-<max>`.
import { SYSTEM, Viewer } from '@photo-sphere-viewer/core';
import '@photo-sphere-viewer/core/index.css';
import type { Maps } from '../map';
import { $, Years, button, drawPlate, outLinks, typing, wirePlan } from '../page';
import {
  DATA,
  FIRST_YEAR,
  LAST_YEAR,
  density,
  fitting,
  format3d,
  hashOf,
  setAddress,
  json,
  parse3d,
  parsePhoto3d,
  shownIn,
  whenLabel,
  yearsLabel,
  yearsTaken,
  type Photo,
  type Place3d,
  type Walk,
} from '../walk';
import { PhotoLayer } from './photo';
import './walk3d.css';
import {
  bearingOf,
  correction,
  nearestTo,
  pointers,
  shownHere,
  stepTarget,
  toward,
  turn,
  yawOf,
  type OldPhoto,
  type Panorama,
  type Scene,
} from './walk3d';

const URL_AFTER_MS = 300; // the link is written once the view rests
const FADE_MS = 700; // from one panorama to the next
const GROUND = -0.45; // radians below the horizon: where the ways on are shown
const HIDDEN_DEG = 70; // a way or a pointer farther than this from the way one looks is hidden

const image = (p: Panorama) => (p.image.startsWith('https://') ? p.image : DATA + p.image);
const fadeOf = () => Number($<HTMLInputElement>('fade').value) / 100;

/** The viewer (none without WebGL: the page then says the street could not be shown). */
function newViewer(): Viewer {
  SYSTEM.load(); // the pixels drawn kept to twice the CSS pixels, as the rest of the site
  SYSTEM.pixelRatio = density();
  return new Viewer({ container: $('view'), navbar: false, keyboard: false, loadingTxt: '', defaultZoomLvl: 30 });
}

class Walk3d {
  private viewer: Viewer;
  private layer: PhotoLayer;
  private scene: Scene;
  private walk: Photo[] = [];
  private years: Years;
  private here!: Panorama;
  private going: number | null = null; // the heading one will face, while the next panorama comes
  private facing = ''; // the photo of the walk one faces
  private maps: Maps | null = null;
  private restTimer = 0;
  private planOpen: () => boolean;

  constructor(viewer: Viewer, scene: Scene, walk: Photo[]) {
    this.scene = scene;
    this.walk = walk;
    this.viewer = viewer;
    this.layer = new PhotoLayer(this.viewer, (photo) => this.imageOf(photo), fadeOf());
    this.years = new Years($('range'), FIRST_YEAR, LAST_YEAR, () => this.showYears());
    this.planOpen = wirePlan(
      () => this.openPlan(),
      () => this.closePlan(),
    );
    this.listen();
  }

  /** The way one looks, as a compass bearing (degrees); while the next panorama comes (the
   *  viewer still turned in the one before), the way one will look. */
  get heading(): number {
    return this.going ?? bearingOf(this.here, this.viewer.getPosition().yaw);
  }

  /** Where one stands and looks, for the maps. */
  get spot(): { lon: number; lat: number; heading: number } {
    return { lon: this.here.lon, lat: this.here.lat, heading: this.heading };
  }

  /** A photo's image as the walk has it, its size the screen's. */
  private imageOf(photo: OldPhoto): string {
    const sizes = this.walk.find((p) => p.id === photo.id)?.images;
    return DATA + (sizes ? fitting(sizes, this.viewer.container.clientWidth * density()) : photo.image);
  }

  /** Where the link says (a place, or a photo: where it was placed), else the first panorama. */
  async start(): Promise<void> {
    if (!(await this.followLink())) await this.go(this.scene.panoramas[0], 0);
  }

  /** Go where the link says; false when it says nowhere this walk knows. */
  private async followLink(): Promise<boolean> {
    const place = parse3d(location.hash);
    if (place) {
      await this.goToPlace(place);
      return true;
    }
    const asked = parsePhoto3d(location.hash);
    const photo = asked && this.scene.photos.find((p) => p.id === asked.id);
    const home = photo && this.scene.panoramas.find((p) => p.id === photo.shown[0].panorama);
    if (!asked || !photo || !home) return false;
    this.years.set(asked.range);
    await this.go(home, photo.shown[0].registration.view.heading);
    return true;
  }

  private async goToPlace({ lon, lat, heading, range }: Place3d): Promise<void> {
    this.years.set(range);
    await this.go(nearestTo(this.scene, lon, lat), heading);
  }

  /** Stand in panorama `p`, looking `heading` (degrees), faded in from where one stood; where
   *  one stood when it cannot be shown. */
  private async go(p: Panorama, heading = this.here ? this.heading : 0): Promise<void> {
    const before = this.here;
    this.layer.clear();
    this.here = p;
    this.going = heading;
    $('pointers').replaceChildren();
    $('ways').replaceChildren();
    $('notice').hidden = true;
    try {
      await this.viewer.setPanorama(image(p), {
        sphereCorrection: correction(p),
        position: { yaw: yawOf(p, heading), pitch: 0 },
        transition: before ? { speed: FADE_MS, rotation: false, effect: 'fade' } : false,
        showLoader: !before,
      });
    } catch (error) {
      if (this.here !== p) return;
      this.going = null;
      if (!before) throw error; // the first: the page says so
      this.here = before; // the viewer still shows it
      $('notice').hidden = false;
      $('notice').textContent = "Cette vue d'aujourd'hui n'a pas pu être chargée.";
    }
    if (this.here !== p && this.here !== before) return; // gone elsewhere meanwhile
    this.going = null;
    this.buildWays();
    this.drawPhotos();
    this.looked(true);
    (window as unknown as { ready: boolean }).ready = true; // polled by tests/visual/walk3d.mjs
  }

  private listen(): void {
    this.viewer.addEventListener('position-updated', () => {
      this.placeOnScreen();
      this.looked(false);
    });
    this.viewer.addEventListener('size-updated', () => this.placeOnScreen());
    this.viewer.addEventListener('zoom-updated', () => this.placeOnScreen());
    this.viewer.addEventListener('click', ({ data }) => {
      const photo = this.layer.at(this.viewer.dataHelper.sphericalCoordsToVector3({ yaw: data.yaw, pitch: data.pitch }));
      if (photo) return this.showCard(photo);
      this.showCard(null);
      if (data.pitch < GROUND / 2) this.step(bearingOf(this.here, data.yaw)); // on the ground
    });
    document.addEventListener('keydown', (e) => {
      if (this.planOpen() || typing(e)) return;
      const { yaw, pitch } = this.viewer.getPosition();
      if (e.key === 'ArrowLeft') this.viewer.rotate({ yaw: yaw - Math.PI / 12, pitch });
      else if (e.key === 'ArrowRight') this.viewer.rotate({ yaw: yaw + Math.PI / 12, pitch });
      else if (e.key === 'ArrowUp') this.step(this.heading);
      else if (e.key === 'Escape') this.showCard(null);
      else if (e.key === 'm' || e.key === 'M') this.openPlan();
    });
    $<HTMLInputElement>('fade').addEventListener('input', () => this.layer.setFade(fadeOf()));
    $('card-close').addEventListener('click', () => this.showCard(null));
    window.addEventListener('hashchange', () => {
      if (location.hash !== this.hash()) void this.followLink();
    });
  }

  private step(bearing: number): void {
    const next = stepTarget(this.scene, this.here, bearing);
    if (next) void this.go(next);
  }

  private hash(): string {
    return format3d({ lon: this.here.lon, lat: this.here.lat, heading: this.heading, range: this.years.range });
  }

  /** The way one looks changed (`stepped`: or where one stands): the photo of the walk one
   *  faces, at once when it changes; the map and the link once the view rests. */
  private looked(stepped: boolean): void {
    const facing = toward(this.walk, this.here, this.heading);
    if (stepped || (facing?.id ?? '') !== this.facing) {
      this.facing = facing?.id ?? '';
      $<HTMLAnchorElement>('back').href = facing ? `./${hashOf(facing.id)}` : './';
      drawPlate(facing);
    }
    window.clearTimeout(this.restTimer);
    this.restTimer = window.setTimeout(() => {
      this.maps?.show(this.spot);
      setAddress(this.hash(), false);
    }, URL_AFTER_MS);
  }

  /** The old photos shown here (all years: the years chosen only show or hide them), and the
   *  years of those here and near marked on the slider. */
  private drawPhotos(): void {
    if (!this.here) return;
    const all: [number, number] = [FIRST_YEAR, LAST_YEAR];
    const here = shownHere(this.scene.photos, this.here.id, all, true);
    this.layer.stand(this.here, here);
    const near = pointers(this.scene, this.here, all, true).map(({ photo }) => photo);
    this.years.mark(yearsTaken([...here.map(([photo]) => photo), ...near]));
    this.showYears();
  }

  /** The photos of the years chosen, here and the way to those near shown elsewhere. */
  private showYears(): void {
    if (!this.here) return;
    const { range, whole } = this.years;
    this.layer.only((photo) => shownIn(photo.years, range, whole));
    $('pointers').replaceChildren(
      ...pointers(this.scene, this.here, range, whole).map(({ photo, home, bearing }) => {
        const b = button('pointer', 'Photo ancienne par là', () => void this.go(home, photo.shown[0].registration.view.heading));
        b.dataset.bearing = String(bearing);
        b.setAttribute('aria-label', `${photo.title} : y aller`);
        return b;
      }),
    );
    this.placeOnScreen();
    this.drawCredits();
  }

  /** The ways on from this panorama, on the ground toward each. */
  private buildWays(): void {
    $('ways').replaceChildren(
      ...this.here.links.map(([index, bearing]) => {
        const b = button('way', '', () => void this.go(this.scene.panoramas[index]));
        b.dataset.bearing = String(bearing);
        b.dataset.ground = 'yes';
        b.setAttribute('aria-label', 'Avancer par là');
        return b;
      }),
    );
  }

  /** The ways and the pointers where their bearing is on screen. */
  private placeOnScreen(): void {
    const looking = this.heading;
    for (const b of [...$('ways').children, ...$('pointers').children] as HTMLElement[]) {
      const bearing = Number(b.dataset.bearing);
      const shown = Math.abs(turn(looking, bearing)) <= HIDDEN_DEG;
      b.hidden = !shown;
      if (!shown) continue;
      const at = this.viewer.dataHelper.sphericalCoordsToViewerCoords({ yaw: yawOf(this.here, bearing), pitch: b.dataset.ground ? GROUND : 0.05 });
      Object.assign(b.style, { left: `${at.x}px`, top: `${at.y}px` });
    }
  }

  private drawCredits(): void {
    const today = document.createElement('p');
    const url = `https://api.panoramax.xyz/#focus=pic&pic=${this.here.id}`;
    today.append(`Rue d'aujourd'hui (${this.here.taken.slice(0, 4)}) : ${this.scene.credits[this.here.credit]}. `, ...outLinks(url, 'Ouvrir dans Panoramax'));
    $('credits-list').replaceChildren(
      today,
      ...this.layer.photos.map((photo) =>
        Object.assign(document.createElement('p'), { textContent: `${yearsLabel(photo.years) || 's. d.'}, ${photo.title}. ${photo.credit}.` }),
      ),
    );
  }

  /** An old photo's card (null: closed): what it is, its credit, and where to see it. */
  private showCard(photo: OldPhoto | null): void {
    $('card').hidden = !photo;
    if (!photo) return;
    $('card-when').textContent = whenLabel(photo.years);
    $('card-title').textContent = photo.title;
    $('card-credit').textContent = photo.credit;
    $('card-links').replaceChildren(Object.assign(document.createElement('a'), { href: `./${hashOf(photo.id)}`, textContent: 'Voir dans la balade' }));
  }

  /** The maps, once the street is shown (they weigh more than the rest). */
  async loadMaps(maps: Promise<typeof import('../map')>): Promise<void> {
    const { Maps } = await maps;
    this.maps = new Maps($('minimap'), this.walk);
    this.maps.show(this.spot);
  }

  private openPlan(): void {
    if (!this.maps) return;
    $('plan').hidden = false;
    this.maps.openPlan($('plan-map'), this.spot, (id) => {
      this.closePlan();
      const placed = this.scene.photos.find((p) => p.id === id);
      const home = placed && this.scene.panoramas.find((p) => p.id === placed.shown[0].panorama);
      const spot = this.walk.find((p) => p.id === id);
      if (home) void this.go(home, placed.shown[0].registration.view.heading); // where it shows
      else if (spot) void this.go(nearestTo(this.scene, spot.lon, spot.lat), spot.heading ?? this.heading);
    });
    $('plan-close').focus();
  }

  private closePlan(): void {
    $('plan').hidden = true;
    $('minimap').focus();
  }
}

Promise.all([json<Scene>('scene.json'), json<Walk>('walk.json')]) // the photos' sizes with the scene
  .then(async ([scene, walk]) => {
    const maps = import('../map'); // fetched alongside the first panorama, built after it
    const walk3d = new Walk3d(newViewer(), scene, walk.photos);
    await walk3d.start();
    await walk3d.loadMaps(maps);
  })
  .catch(() => {
    document.body.append(
      Object.assign(document.createElement('p'), { id: 'failed', textContent: "La rue n'a pas pu être chargée. Rechargez la page dans un instant." }),
    );
  });


