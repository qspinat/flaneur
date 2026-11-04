/** The maps: a small one beside the photograph (where I stand, which way I look) and the
 *  plan of every photograph, full screen. Background: Plan IGN (Géoplateforme, Etalab 2.0). */
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
      maxzoom: 19,
      attribution: '<a href="https://www.ign.fr/">IGN</a> Plan IGN',
    },
  },
  layers: [{ id: 'ign', type: 'raster', source: 'ign' }],
};

const ENAMEL = '#1b3a6b';
const RIM = '#2f6b4f';
const CONE_M = 40;
const CONE_HALF_DEG = 30;
const TAP_PX = 22; // how far from a dot a tap still reaches it

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
  map.addLayer({ id: 'cone', type: 'fill', source: 'cone', paint: { 'fill-color': ENAMEL, 'fill-opacity': 0.35 } });
  map.addLayer({
    id: 'photos',
    type: 'circle',
    source: 'photos',
    paint: {
      'circle-radius': ['case', ['get', 'here'], 8, 6],
      'circle-color': ['case', ['get', 'here'], ENAMEL, RIM],
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
      // the dot under the finger (a tap beside every dot goes nowhere)
      const { x, y } = event.point;
      const touched = plan.queryRenderedFeatures([[x - TAP_PX, y - TAP_PX], [x + TAP_PX, y + TAP_PX]], { layers: ['photos'] });
      const nearest = touched
        .map((f) => {
          const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
          const p = plan.project([lon, lat]);
          return { id: f.properties.id as string, d: Math.hypot(p.x - x, p.y - y) };
        })
        .sort((a, b) => a.d - b.d)[0];
      if (nearest) go(nearest.id);
    });
  }
}
