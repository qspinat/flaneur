// Shared by the local pages that place photos on a map (see localpage.page): IGN's plan and
// its 1949 aerial photographs, a camera's wedge, the years of a photo.

const IGN = (layer, format, matrixSet = 'PM') =>
  'https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&STYLE=normal' +
  `&LAYER=${layer}&TILEMATRIXSET=${matrixSet}&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}` +
  `&FORMAT=${format}`;

/** IGN's plan, with its 1949 aerial photographs hidden under it (see toggleAerial), and a
 *  `cone` source for the camera's wedge drawn as a layer; `sources` and `layers` added on top. */
function plainStyle(sources = {}, layers = []) {
  return {
    version: 8,
    sources: {
      plan: { type: 'raster', tileSize: 256, maxzoom: 19,
              tiles: [IGN('GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2', 'image/png')],
              attribution: 'IGN — Plan IGN' },
      aerial: { type: 'raster', tileSize: 256, minzoom: 6, maxzoom: 18,
                tiles: [IGN('ORTHOIMAGERY.EDUGEO.PARIS1949', 'image/png', 'PM_6_18')],
                attribution: 'IGN — Paris 1949' },
      cone: { type: 'geojson', data: { type: 'FeatureCollection', features: [] } },
      ...sources,
    },
    layers: [
      { id: 'plan', type: 'raster', source: 'plan' },
      { id: 'aerial', type: 'raster', source: 'aerial', layout: { visibility: 'none' } },
      { id: 'cone', type: 'fill', source: 'cone',
        paint: { 'fill-color': '#8a3b12', 'fill-opacity': 0.35 } },
      ...layers,
    ],
  };
}

/** The plan or the 1949 aerial photographs, one for the other. */
function toggleAerial(map) {
  const visible = map.getLayoutProperty('aerial', 'visibility') === 'visible';
  map.setLayoutProperty('aerial', 'visibility', visible ? 'none' : 'visible');
}

/** How many metres a degree of longitude and of latitude span at `lat`. */
function metresPerDegree(lat) {
  return [111320 * Math.cos(lat * Math.PI / 180), 111320];
}

/** How far apart two points ([lon, lat]) stand, in metres (near one another). */
function metresBetween(a, b) {
  const [east, north] = metresPerDegree(b[1]);
  return Math.hypot((a[0] - b[0]) * east, (a[1] - b[1]) * north);
}

/** A wedge from the camera, `heading` degrees from north, `halfAngle` either side, `length`
 *  metres long, as a GeoJSON polygon. */
function cone(lon, lat, heading, halfAngle, length) {
  const [metresPerDegreeLon, metresPerDegreeLat] = metresPerDegree(lat);
  const ring = [[lon, lat]];
  for (let a = heading - halfAngle; a <= heading + halfAngle; a += 5) {
    const r = a * Math.PI / 180;
    ring.push([lon + length * Math.sin(r) / metresPerDegreeLon,
               lat + length * Math.cos(r) / metresPerDegreeLat]);
  }
  ring.push([lon, lat]);
  return { type: 'Feature', geometry: { type: 'Polygon', coordinates: [ring] } };
}

/** A photo's years, as after its title: " (1900–1910)", or nothing. */
function yearsOf(years) {
  const known = [...new Set((years || []).filter((y) => y != null))];
  return known.length ? ` (${known.join('–')})` : '';
}
