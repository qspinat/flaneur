#!/bin/sh
# Publish the site on GitHub Pages: the build as a single commit on the gh-pages branch of the
# repository's origin, replacing the last one (what it already holds is fetched first, so
# images that did not change are not sent again).
set -eu
cd "$(dirname "$0")/.."
remote=$(git remote get-url origin)
npm run -s build
touch dist/.nojekyll # served as is, no Jekyll
cd dist
rm -rf .git
git init -q -b gh-pages
git fetch -q --depth 1 "$remote" gh-pages 2>/dev/null || true # nothing yet at the first publication
git add -A
git -c user.name="Quentin Spinat" -c user.email=qs-dev@tuta.com commit -q -m "Site"
git push -q -f "$remote" gh-pages
rm -rf .git
echo "publié : $(git -C .. log -1 --format=%h) → gh-pages"
