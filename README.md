# Flâneur

Marcher dans le Paris d'autrefois, une photographie d'archive après l'autre, comme dans
Street View : chaque photo sait d'où elle a été prise et dans quelle direction, ce qui permet
d'avancer dans la rue, de tourner, de voir le même endroit à d'autres époques — et aujourd'hui.

## Règle des droits

Une image n'entre sur le site que si elle est dans le domaine public ou sous licence ouverte ;
sa licence et sa mention de source sont stockées avec elle et affichées sous elle.

**Le site reste non commercial** (ni publicité, ni offre payante) : les images issues de
Gallica / BnF ne sont libres que pour un usage non commercial. Un usage commercial demanderait
une licence BnF.

## Organisation

- `pipeline/` — Python (uv) : récolte des sources, lecture des légendes, placement des
  photos (position + direction), graphe de navigation, export statique.
- `web/` — TypeScript + Vite : le site, entièrement statique.
- `data/overrides/` — corrections faites à la main (commitées) ; le reste de `data/` est
  reconstruit par le pipeline.

## Démarrer

```bash
cd pipeline && cp .env.example .env   # puis renseigner PARIS_MUSEES_TOKEN
uv run flaneur --help
uv run pytest

cd web && npm install && npm run dev
```

## Données et crédits

Fond de carte et données : IGN (Géoplateforme, Licence Ouverte Etalab 2.0). Noms de voies :
Ville de Paris (ODbL). Vues actuelles : Panoramax. Les positions exportées sont publiées sous
ODbL.
