/** The maps: the plan the site opens on (every photograph, turned the way it looked), a small
 *  one beside a photograph (where I stand, which way I look), and the 3D walk's full-screen
 *  plan. Background: Plan IGN (Géoplateforme, Etalab 2.0), faded onto paper. */
import maplibregl, { type GeoJSONSource, type Map as MapLibre } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
/** A photo on the map. */
export interface Dot {
  id: string;
  lon: number;
  lat: number;
}

/** Where I stand and which way I look; `id`: the photo there, marked. */
export interface Spot {
  lon: number;
  lat: number;
  heading: number | null;
  id?: string;
}

const PLAN_IGN =
  'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&STYLE=normal' +
  '&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}' +
  '&TILECOL={x}&FORMAT=image/png';

const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    ign: {
      type: 'raster',
      tiles: [PLAN_IGN],
      tileSize: 256,
      maxzoom: 18, // its level 19 covers only parts of Paris: closer, level 18 is enlarged
      attribution: '<a href="https://www.ign.fr/">IGN</a> Plan IGN',
    },
  },
  // the plan faded onto albumen paper, like an old print; what is drawn on it keeps its colours
  layers: [
    { id: 'paper', type: 'background', paint: { 'background-color': '#e8dcc4' } },
    {
      id: 'ign',
      type: 'raster',
      source: 'ign',
      paint: { 'raster-opacity': 0.78, 'raster-saturation': -0.75, 'raster-contrast': -0.1 },
    },
  ],
};

const ENAMEL = '#1b3a6b';
const RIM = '#2f6b4f';
const CONE_M = 40;
const CONE_HALF_DEG = 30;
const TAP_PX = 22; // how far from a dot a tap still reaches it

/** The photo whose mark is nearest a tap at `point` on `layer`, within TAP_PX; undefined
 *  when none is that near. */
function nearestAt(map: MapLibre, point: maplibregl.Point, layer: string): string | undefined {
  const { x, y } = point;
  return map
    .queryRenderedFeatures([[x - TAP_PX, y - TAP_PX], [x + TAP_PX, y + TAP_PX]], { layers: [layer] })
    .map((f) => {
      const p = map.project((f.geometry as GeoJSON.Point).coordinates as [number, number]);
      return { id: f.properties.id as string, d: Math.hypot(p.x - x, p.y - y) };
    })
    .sort((a, b) => a.d - b.d)[0]?.id;
}

/** A sector of a circle, in degrees, for the direction the camera faced. */
function cone(lon: number, lat: number, heading: number): GeoJSON.Feature {
  const metresPerLat = 111_200;
  const metresPerLon = metresPerLat * Math.cos((lat * Math.PI) / 180);
  const ring: [number, number][] = [[lon, lat]];
  for (let a = heading - CONE_HALF_DEG; a <= heading + CONE_HALF_DEG; a += 5) {
    const r = (a * Math.PI) / 180;
    ring.push([lon + (CONE_M * Math.sin(r)) / metresPerLon, lat + (CONE_M * Math.cos(r)) / metresPerLat]);
  }
  ring.push([lon, lat]);
  return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [ring] } };
}

function points(photos: Dot[], here?: string): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: photos.map((p) => ({
      type: 'Feature',
      properties: { id: p.id, here: p.id === here },
      geometry: { type: 'Point', coordinates: [p.lon, p.lat] },
    })),
  };
}

function addLayers(map: MapLibre, photos: Dot[], here?: string): void {
  map.addSource('photos', { type: 'geojson', data: points(photos, here) });
  map.addSource('cone', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({ id: 'cone', type: 'fill', source: 'cone', paint: { 'fill-color': RIM, 'fill-opacity': 0.35 } });
  map.addLayer({
    id: 'photos',
    type: 'circle',
    source: 'photos',
    paint: {
      'circle-radius': ['case', ['get', 'here'], 8, 6],
      'circle-color': ['case', ['get', 'here'], RIM, ENAMEL], // as on the plan: the photo here in green
      'circle-stroke-color': '#fff',
      'circle-stroke-width': 1.5,
    },
  });
}

export class Maps {
  private mini: MapLibre;
  private plan: MapLibre | null = null;
  private photos: Dot[];
  private marked: string | undefined; // the photo marked on the minimap now
  private centre = '';

  constructor(miniContainer: HTMLElement, photos: Dot[]) {
    this.photos = photos;
    this.mini = new maplibregl.Map({
      container: miniContainer,
      style: STYLE,
      center: [photos[0].lon, photos[0].lat],
      zoom: 17,
      interactive: false,
      attributionControl: { compact: true },
    });
    this.mini.on('load', () => addLayers(this.mini, photos));
  }

  /** Show where I stand and which way I look: the marked photo is redrawn only when it
   * changes, and the map moves only when I do. */
  show(spot: Spot): void {
    const waiting = this.next !== null;
    this.next = spot; // before the map is ready, only the latest place counts
    if (!waiting) this.whenReady(() => this.update());
  }

  private next: Spot | null = null;

  private whenReady(then: () => void): void {
    if (this.mini.getSource('photos')) then();
    else this.mini.once('load', then);
  }

  private update(): void {
    const spot = this.next!;
    this.next = null;
    if (spot.id !== this.marked) {
      this.marked = spot.id;
      (this.mini.getSource('photos') as GeoJSONSource).setData(points(this.photos, spot.id));
    }
    (this.mini.getSource('cone') as GeoJSONSource).setData({
      type: 'FeatureCollection',
      features: spot.heading == null ? [] : [cone(spot.lon, spot.lat, spot.heading)],
    });
    const centre = `${spot.lon},${spot.lat}`;
    if (centre !== this.centre) {
      this.centre = centre;
      this.mini.easeTo({ center: [spot.lon, spot.lat], duration: 400 });
    }
  }

  /** The plan of every photograph; `go` is called with the photo nearest a tap. */
  openPlan(container: HTMLElement, here: Spot, go: (id: string) => void): void {
    if (this.plan) {
      this.plan.resize();
      this.plan.jumpTo({ center: [here.lon, here.lat] });
      (this.plan.getSource('photos') as GeoJSONSource | undefined)?.setData(points(this.photos, here.id));
      return;
    }
    const plan = new maplibregl.Map({ container, style: STYLE, center: [here.lon, here.lat], zoom: 17.5 });
    this.plan = plan;
    plan.on('load', () => addLayers(plan, this.photos, here.id));
    plan.on('click', (event) => {
      const id = nearestAt(plan, event.point, 'photos'); // a tap beside every dot goes nowhere
      if (id) go(id);
    });
  }
}

/** A photo on the plan: where its camera stood and which way it looked. */
export type Viewpoint = Dot & { heading: number | null };

/** The mark of a photo on the plan, as old maps drew a viewpoint: a wedge opening the way the
 *  camera looked (drawn pointing up; the map turns it), from a round foot where it stood. */
function viewpointIcon(size: number, fill: string, edge: string, wedge = true): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const g = canvas.getContext('2d')!;
  const c = size / 2;
  g.lineJoin = 'round';
  g.lineWidth = size * 0.06;
  g.strokeStyle = edge;
  g.fillStyle = fill;
  if (wedge) {
    g.beginPath();
    g.moveTo(c, c);
    g.arc(c, c, c - g.lineWidth, -Math.PI / 2 - 0.5, -Math.PI / 2 + 0.5);
    g.closePath();
    g.fill();
    g.stroke();
  }
  g.beginPath();
  g.arc(c, c, size * (wedge ? 0.12 : 0.18), 0, 2 * Math.PI);
  g.fill();
  g.stroke();
  return g.getImageData(0, 0, size, size);
}

function viewpoints(photos: Viewpoint[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: photos.map((p) => ({
      type: 'Feature',
      properties: { id: p.id, heading: p.heading ?? 0, turned: p.heading != null }, // unknown: a foot alone
      geometry: { type: 'Point', coordinates: [p.lon, p.lat] },
    })),
  };
}

/** Marks turned the way their camera looked (a foot alone when that is not known), all drawn
 *  however close; `image`: which mark. */
function turned(image: string): maplibregl.SymbolLayerSpecification['layout'] {
  return {
    'icon-image': ['case', ['get', 'turned'], image, `${image}-foot`],
    'icon-rotate': ['get', 'heading'],
    'icon-rotation-alignment': 'map',
    'icon-allow-overlap': true,
    'icon-ignore-placement': true,
    'icon-size': ['interpolate', ['linear'], ['zoom'], 14, 0.45, 17, 0.8, 19, 1.1],
  };
}

/** The plan the site opens on: every photograph where its camera stood, turned the way it
 *  looked. A tap on one (or near it) calls `choose` with its id; a tap elsewhere, with null.
 *  All the marks are laid once; the one chosen is drawn apart, over them. */
export class Plan {
  private map: MapLibre;
  private pending: Viewpoint | null = null; // chosen before the plan was ready
  private filter: maplibregl.FilterSpecification | null = null; // the years shown (null: all)

  constructor(container: HTMLElement, photos: Viewpoint[], choose: (id: string | null) => void, padding: maplibregl.PaddingOptions) {
    const lons = photos.map((p) => p.lon);
    const lats = photos.map((p) => p.lat);
    this.map = new maplibregl.Map({
      container,
      style: STYLE,
      center: [(Math.min(...lons) + Math.max(...lons)) / 2, (Math.min(...lats) + Math.max(...lats)) / 2],
      zoom: 15,
      maxZoom: 19.5,
      minZoom: 13,
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
    });
    this.map.touchZoomRotate.disableRotation();
    // every photograph in view, in what the overlays (the plate, the welcome) leave clear
    this.map.setPadding(padding);
    this.map.fitBounds(
      [
        [Math.min(...lons), Math.min(...lats)],
        [Math.max(...lons), Math.max(...lats)],
      ],
      { animate: false },
    );
    // the marks as soon as the style is read, without waiting for every tile of the plan
    this.map.once('style.load', () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      for (const [name, size, fill] of [['view', 44, ENAMEL], ['view-here', 70, RIM]] as const) {
        this.map.addImage(name, viewpointIcon(Math.round(size * ratio), fill, '#f5efe3'), { pixelRatio: ratio });
        this.map.addImage(`${name}-foot`, viewpointIcon(Math.round(size * ratio), fill, '#f5efe3', false), { pixelRatio: ratio });
      }
      this.map.addSource('views', { type: 'geojson', data: viewpoints(photos) });
      this.map.addSource('chosen', { type: 'geojson', data: viewpoints(this.pending ? [this.pending] : []) });
      this.map.addLayer({ id: 'views', type: 'symbol', source: 'views', layout: turned('view'), ...(this.filter && { filter: this.filter }) });
      this.map.addLayer({
        id: 'chosen-ring', // a soft ring under the photograph chosen
        type: 'circle',
        source: 'chosen',
        paint: { 'circle-radius': 22, 'circle-color': RIM, 'circle-opacity': 0.18, 'circle-stroke-color': RIM, 'circle-stroke-width': 2 },
      });
      this.map.addLayer({ id: 'chosen', type: 'symbol', source: 'chosen', layout: turned('view-here') });
    });
    this.map.on('click', (event) => this.map.getLayer('views') && choose(nearestAt(this.map, event.point, 'views') ?? null));
    this.map.on('mousemove', (event) => {
      if (this.map.isMoving() || event.originalEvent.buttons || !this.map.getLayer('views')) return;
      const over = nearestAt(this.map, event.point, 'views') !== undefined;
      this.map.getCanvas().style.cursor = over ? 'pointer' : '';
    });
  }

  /** Mark one photograph (null: none). */
  select(photo: Viewpoint | null): void {
    const chosen = this.map.getSource('chosen') as GeoJSONSource | undefined;
    if (chosen) chosen.setData(viewpoints(photo ? [photo] : []));
    else this.pending = photo; // laid with the marks, when the style is read
  }

  /** Bring a photograph into view, in what the overlays leave clear (`padding`). */
  reveal(photo: Viewpoint, padding: maplibregl.PaddingOptions): void {
    this.map.easeTo({ center: [photo.lon, photo.lat], zoom: Math.max(this.map.getZoom(), 17.5), padding, duration: 500 });
  }

  /** Only these photographs (null: all of them). */
  only(ids: string[] | null): void {
    const filter: maplibregl.FilterSpecification | null = ids && ['in', ['get', 'id'], ['literal', ids]];
    this.filter = filter;
    if (this.map.getLayer('views')) this.map.setFilter('views', filter);
  }

  /** After the plan was hidden: its size may have changed meanwhile. */
  shown(): void {
    this.map.resize();
  }
}
