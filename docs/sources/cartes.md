# Cartes anciennes de Paris

Usage : **interne** (placer les photos, outil de revue), pas en fond du site public (choix
de conception). Elles servent surtout aux rues disparues, que les vues actuelles ne
montrent plus.

Chaque ligne dit si l'accès a été **testé** (une tuile de Paris téléchargée *et regardée* :
une tuile d'erreur peut arriver avec un code HTTP 200).

## Tuiles prêtes à l'emploi (IGN Géoplateforme, sans clé)

Toutes en WMTS sur `https://data.geopf.fr/wmts` — `SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&STYLE=normal&LAYER=…&TILEMATRIXSET=…&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=…`.
Licence Ouverte Etalab 2.0 (mention « IGN » + date de mise à jour) pour le Géoportail ;
celle des couches EDUGEO reste à confirmer avant tout usage public.

| Époque | Couche (`LAYER`) | `TILEMATRIXSET` | Format | Test 28/10/2026 |
|---|---|---|---|---|
| 1818–1824 | `GEOGRAPHICALGRIDSYSTEMS.ETATMAJOR10` (état-major, environs de Paris, 1:10 000) | `PM_6_16` | jpeg | ✅ tuile reçue |
| 1820–1866 | `GEOGRAPHICALGRIDSYSTEMS.ETATMAJOR40` (état-major 1:40 000) | `PM_6_15` | jpeg | ✅ tuile reçue |
| 1949 | `ORTHOIMAGERY.EDUGEO.PARIS1949` (photo aérienne) | `PM_6_18` | png | ✅ regardée : rues et toits nets au zoom 16 |
| 1950–1965 | `ORTHOIMAGERY.ORTHOPHOTOS.1950-1965` | `PM_0_18` | png | ✅ tuile reçue |
| 1964 | `GEOGRAPHICALGRIDSYSTEMS.EDUGEO.PARIS1964` (carte) | `PM_6_16` | png | ⚠️ tuile de 852 o, probablement vide à cet endroit |
| 1965 | `ORTHOIMAGERY.EDUGEO.PARIS1965` (photo aérienne) | `PM_6_16` | png | non testé |
| XVIIIe | `BNF-IGNF_GEOGRAPHICALGRIDSYSTEMS.CASSINI` | `PM_6_14` | png | non testé (échelle trop petite pour une rue) |
| aujourd'hui | `GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2` | `PM` | png | ✅ ; en vecteur : `data.geopf.fr/tms/1.0.0/PLAN.IGN/{z}/{x}/{y}.pbf` |

## Plans de Paris géoréférencés (SoDUCo, annotations Allmaps)

Liste : <https://soduco.geohistoricaldata.org/AllmapsAnnotations/>. Chaque annotation
contient, par feuille, l'URL de l'image et ~9 points de calage (pixel ↔ coordonnées).

| Plan | Année | Feuilles | Annotation |
|---|---|---|---|
| Atlas de Verniquet (Gallica) | 1790 | 72 | `https://dataverse.harvard.edu/api/access/datafile/7002730` |
| Atlas de Verniquet (David Rumsey) | 1790 | ? | `https://dataverse.harvard.edu/api/access/datafile/7002615` |
| Atlas municipal (Ville de Paris, BHdV) | 1868, 1878, 1886, 1887, 1888, 1900, 1912, 1925, 1937 | 16 (1868) | `https://raw.githubusercontent.com/soduco/allmaps_annotations/main/output/bhdv_atlas_municipal_<année>/annotation_bhdv_atlas_municipal_<année>.json` |

**Test du 28/10/2026 : ❌ inutilisables ce jour.** Les tuiles `allmaps.xyz` répondent 200
mais contiennent une image d'erreur : le serveur d'images de SoDUCo
(`iiif.geohistoricaldata.org`) ne répond pas, et Gallica renvoie 403 à Allmaps et 500 en
accès direct. À retester. Plan de repli si ça dure : télécharger les feuilles quand un
serveur répond, les caler nous-mêmes avec les points de l'annotation (GDAL), servir les
tuiles en local.

Licences : Gallica = réutilisation non commerciale libre avec mention « Source
gallica.bnf.fr / BnF » ; annotations SoDUCo CC BY ; images de l'Atlas municipal (BHdV) :
à vérifier.

## Atlas de Jacoubet (1827–1836) — la carte du Paris d'avant Haussmann

**Le meilleur fond pour les rues disparues** (Marville photographie en 1865–1868 des rues
que Jacoubet a levées trente ans plus tôt) : 54 feuilles au 1:2000, chaque rue nommée.

- Dépôt : <https://github.com/soduco/Atlas-de-Jacoubet> (SoDUCo) — scans BHVP (~14 Mo par
  feuille, ~700 Mo en tout), points de calage par feuille (`gcps/*.points`, format QGIS),
  lignes de découpe (`cutlines/`), scripts GDAL pour produire un VRT puis des MBTiles.
- Source : Bibliothèque historique de la Ville de Paris, FM AT 11. Licence des scans :
  à vérifier (usage interne d'abord).
- **Test du 28/10/2026 : ✅ accessible** (API GitHub, arborescence et points listés).
- Usage prévu : caler les feuilles du quartier pilote en local (GDAL), pour l'outil de
  revue et pour lire les noms de rues autour d'une photo de rue disparue.

## Données vectorielles (pour le géocodage historique)

| Quoi | Époque | Où | Licence |
|---|---|---|---|
| Dénominations caduques des voies (ancien nom → historique) | toutes | <https://opendata.paris.fr/explore/dataset/denominations-des-voies-caduques/> | ODbL |
| Emprises des voies actuelles (polygones + anciens noms) | aujourd'hui | <https://opendata.paris.fr/explore/dataset/denominations-emprises-voies-actuelles/> | ODbL |
| ALPAGE – cadastre Vasserot (parcelles, voies, adresses) | 1810–1836 | Stanford EarthWorks : <https://purl.stanford.edu/bz020fp6123> | ODbL |
| Réseaux de rues vectorisés (Verniquet, Jacoubet, Poubelle…) | 1790–1888 | <https://geohistoricaldata.github.io> | CC BY (à confirmer) |
| JADIS – plans 1760–1949 alignés + rues vectorisées | 1760–1949 | <https://bnf-jadis.github.io> | CC BY 2.0 |
| Annuaires parisiens géocodés (SoDUCo) | 1798–1914 | <https://nakala.fr/10.34847/nkl.98eem49t> | à vérifier |

## Pour mémoire (pas d'accès réutilisable)

- APUR, Cassini Grand Paris : 19 plans de 1728 à aujourd'hui, visionneuse seulement.
- David Rumsey Georeferencer : Verniquet, compte requis, CC BY-NC-SA.
- Plan de Turgot (1734–1739) : aucune version calée trouvée.
