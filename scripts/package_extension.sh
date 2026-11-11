#!/usr/bin/env bash
# Builds the Chrome extension for THIS deployment, twice, into the web app's
# static dir:
#
#   fred-extension.zip        offered in Settings → Browser extension, to unzip
#                             and load unpacked; the manifest sits under extension/
#   fred-extension-store.zip  what the Chrome Web Store console accepts: the
#                             manifest MUST be at the zip root (docs/EXTENSION.md)
#
# The extension in extension/ carries a placeholder site (https://aplec.example.org);
# the build replaces it with the origin of FRED_LIVE_WEB_BASE_URL, taken from the
# environment or from .env.worker, in the host permissions and as the default
# server in Options. Rerun after changing the site URL or anything in extension/.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
STATIC="fred_live_web/static"
PLACEHOLDER="https://aplec.example.org"

SITE="${FRED_LIVE_WEB_BASE_URL:-}"
if [[ -z "$SITE" && -f .env.worker ]]; then
  SITE="$(grep -E '^FRED_LIVE_WEB_BASE_URL=' .env.worker | tail -1 | cut -d= -f2- | tr -d "\"' ")"
fi
ORIGIN="$(python3 - "$SITE" <<'PY'
import sys
from urllib.parse import urlsplit
parts = urlsplit(sys.argv[1].strip())
if parts.scheme not in ("http", "https") or not parts.netloc:
    sys.exit("")
print(f"{parts.scheme}://{parts.netloc}")
PY
)" || true
if [[ -z "$ORIGIN" ]]; then
  echo "error: set FRED_LIVE_WEB_BASE_URL (in .env.worker or the environment) to your Aplec site, e.g. https://aplec.example.org" >&2
  exit 1
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cp -r extension "$BUILD/extension"
python3 - "$BUILD/extension" "$PLACEHOLDER" "$ORIGIN" <<'PY'
import json
import pathlib
import sys

root, placeholder, origin = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
hosts = [h for h in manifest["host_permissions"] if h != placeholder + "/*"]
if origin + "/*" not in hosts:
    hosts.append(origin + "/*")
manifest["host_permissions"] = hosts
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for name in ("options.js", "options.html"):
    path = root / name
    text = path.read_text(encoding="utf-8")
    if placeholder not in text:
        sys.exit(f"{name}: placeholder {placeholder} not found")
    path.write_text(text.replace(placeholder, origin), encoding="utf-8")
PY

VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' extension/manifest.json)"
rm -f "$STATIC/fred-extension.zip" "$STATIC/fred-extension-store.zip"
(cd "$BUILD" && zip -qr "$ROOT_DIR/$STATIC/fred-extension.zip" extension -x "extension/.*")
(cd "$BUILD/extension" && zip -qr "$ROOT_DIR/$STATIC/fred-extension-store.zip" . -x '.*' -x '*/.*')

# Fail loudly rather than handing the store a zip it will reject.
unzip -l "$STATIC/fred-extension-store.zip" | grep -qE ' manifest\.json$' || {
  echo "error: manifest.json is not at the root of fred-extension-store.zip" >&2
  exit 1
}

# Version marker next to the zips: installed extensions read it and show a NEW
# badge when the build the site offers is newer than theirs.
printf '{"version": "%s", "packaged_at": "%s"}\n' "$VERSION" "$(date -u +%FT%TZ)" \
  > "$STATIC/fred-extension-version.json"

echo "packaged v$VERSION for $ORIGIN:"
echo "  $STATIC/fred-extension.zip        ($(du -h "$STATIC/fred-extension.zip" | cut -f1))  Settings download, load unpacked"
echo "  $STATIC/fred-extension-store.zip  ($(du -h "$STATIC/fred-extension-store.zip" | cut -f1))  Chrome Web Store upload"
