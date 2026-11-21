#!/usr/bin/env bash
# Deploy the Signdex web app (index.html + debug.html) to the Vultr server with Nginx + Let's Encrypt.
#
#   bash deploy/deploy.sh                 # everything; asks before each command that runs on the server
#   bash deploy/deploy.sh --yes           # same, no prompts (redeploy in one command)
#   bash deploy/deploy.sh build upload    # only some steps (in the order given)
#   LE_EMAIL=you@example.com bash deploy/deploy.sh cert   # optional Let's Encrypt expiry-notice email
#   SIGNDEX_HOST=203.0.113.7 bash deploy/deploy.sh        # another server (default 149.28.32.34)
#
# Steps: dns build setup upload nginx cert nginx verify   (all re-runnable; nginx runs twice on a fresh
# server: HTTP-only first so certbot can answer the challenge, then the HTTPS config once the cert exists)
# Run from Git Bash (Windows) or any POSIX shell with ssh, tar, nslookup and npm.

set -euo pipefail

HOST="${SIGNDEX_HOST:-149.28.32.34}"   # server IP; override with SIGNDEX_HOST=...
SSH_USER=root
KEY="$HOME/.ssh/id_ed25519"
DOMAIN=signdex.study
DOMAINS=("$DOMAIN" "www.$DOMAIN")
APP_DIR=/var/www/signdex          # releases/<timestamp>, current -> newest release
ACME_DIR=/var/www/letsencrypt
KEEP_RELEASES=3

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/web/dist"
YES=0
STEPS=()
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) STEPS+=("$a") ;;
  esac
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(dns build setup upload nginx cert nginx verify)

SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$SSH_USER@$HOST")
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# Run a bash script on the server, after showing it and asking (unless --yes). Extra stdin can be piped in
# by passing a local command as $3 (used by upload to stream the tarball).
remote() {
  local title="$1" script="$2" feed="${3:-}"
  say "server: $title"
  printf '%s\n' "$script" | sed 's/^/  | /'
  if [ "$YES" -ne 1 ]; then
    read -r -p "Run this on $SSH_USER@$HOST? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "skipped"; return 1; }
  fi
  if [ -n "$feed" ]; then
    # script travels as an argument, stdin carries the data
    eval "$feed" | "${SSH[@]}" "bash -c $(printf '%q' "set -euo pipefail; $script")"
  else
    printf 'set -euo pipefail\n%s\n' "$script" | "${SSH[@]}" 'bash -s'
  fi
}

step_dns() {
  say "local: DNS for ${DOMAINS[*]} must resolve to $HOST"
  local tries=0
  while :; do
    local ok=1
    for h in "${DOMAINS[@]}"; do
      if nslookup "$h" 2>/dev/null | sed -n '/^Name:/,$p' | grep -q "$HOST"; then echo "  $h -> $HOST"
      else echo "  $h does not resolve to $HOST yet"; ok=0; fi
    done
    [ "$ok" -eq 1 ] && return 0
    tries=$((tries + 1))
    [ "$tries" -ge 30 ] && { echo "DNS still not pointing at $HOST after 30 min; giving up"; exit 1; }
    echo "  waiting 60 s ($tries/30)"; sleep 60
  done
}

step_build() {
  say "local: build web app"
  (cd "$REPO/web" && npm run build)
  for f in index.html debug.html model/model.onnx model/labels.json mediapipe; do
    [ -e "$DIST/$f" ] || { echo "missing $DIST/$f (model files come from model/ via scripts/copy-model.mjs)"; exit 1; }
  done
  du -sh "$DIST"
}

step_setup() {
  remote "install nginx + certbot, open the firewall for HTTP/HTTPS" "
pkgs=''
for p in nginx certbot; do dpkg -s \$p >/dev/null 2>&1 || pkgs=\"\$pkgs \$p\"; done
if [ -n \"\$pkgs\" ]; then
  apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q \$pkgs
else
  echo 'nginx and certbot already installed'
fi
if command -v ufw >/dev/null && ufw status | grep -q 'Status: active'; then ufw allow 'Nginx Full'; fi
mkdir -p $APP_DIR/releases $ACME_DIR
systemctl enable --now nginx
nginx -v"
}

step_upload() {
  [ -f "$DIST/index.html" ] || { echo "no build in $DIST; run the build step first"; exit 1; }
  local rel; rel="$(date -u +%Y%m%d-%H%M%S)"
  remote "upload web/dist as release $rel, switch $APP_DIR/current to it, keep the newest $KEEP_RELEASES" "
d=$APP_DIR/releases/$rel
mkdir -p \$d
tar -xzf - -C \$d --no-same-owner
find \$d -name .gitkeep -delete
chmod -R a+rX \$d
ln -sfn \$d $APP_DIR/current.tmp && mv -Tf $APP_DIR/current.tmp $APP_DIR/current
ls -1dt $APP_DIR/releases/*/ | tail -n +$((KEEP_RELEASES + 1)) | xargs -r rm -rf
echo \"current -> \$(readlink $APP_DIR/current)\"; du -sh \$d" \
    "tar -C $(printf '%q' "$DIST") -czf - ."
}

nginx_conf() {  # $1 = http | https
  local live=/etc/letsencrypt/live/$DOMAIN
  cat <<EOF
# Managed by deploy/deploy.sh in the signseek repo. Edits here are overwritten on the next deploy.
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAINS[*]};
    location /.well-known/acme-challenge/ { root $ACME_DIR; }
EOF
  if [ "$1" = http ]; then
    cat <<EOF
    root $APP_DIR/current;
    index index.html;
    location / { try_files \$uri \$uri/ =404; }
}
EOF
    return
  fi
  cat <<EOF
    location / { return 301 https://$DOMAIN\$request_uri; }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name www.$DOMAIN;
    ssl_certificate     $live/fullchain.pem;
    ssl_certificate_key $live/privkey.pem;
    return 301 https://$DOMAIN\$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate     $live/fullchain.pem;
    ssl_certificate_key $live/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_cache shared:SSL:10m;

    root $APP_DIR/current;
    index index.html;

    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;
    add_header Permissions-Policy "camera=(self), microphone=(), geolocation=()" always;

    gzip on;
    gzip_types application/javascript application/json application/wasm text/css image/svg+xml;
    gzip_min_length 1024;

    # Vite's hashed bundles never change; everything else (html, model, mediapipe) revalidates via ETag
    location /assets/ { expires 1y; try_files \$uri =404; }
    location / { expires -1; try_files \$uri \$uri/ =404; }
}
EOF
}

step_nginx() {
  # The server picks: HTTPS config if the cert exists, else HTTP-only (so certbot can answer the challenge).
  remote "write the Nginx site for ${DOMAINS[*]} (HTTPS if the cert exists, else HTTP-only), test, reload" "
if [ -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem ]; then
  mode=https
  cat > /etc/nginx/sites-available/signdex <<'NGINX'
$(nginx_conf https)
NGINX
else
  mode=http
  cat > /etc/nginx/sites-available/signdex <<'NGINX'
$(nginx_conf http)
NGINX
fi
ln -sfn /etc/nginx/sites-available/signdex /etc/nginx/sites-enabled/signdex
nginx -t
systemctl reload nginx
echo \"nginx: \$mode config active\""
}

step_cert() {
  local who
  if [ -n "${LE_EMAIL:-}" ]; then who="-m $(printf '%q' "$LE_EMAIL")"; else who="--register-unsafely-without-email"; fi
  remote "get/renew the Let's Encrypt cert for ${DOMAINS[*]} (webroot; agrees to the LE terms; skips if not near expiry)" "
certbot certonly --webroot -w $ACME_DIR $(printf -- '-d %s ' "${DOMAINS[@]}") \\
  --non-interactive --agree-tos $who --keep-until-expiring \\
  --deploy-hook 'systemctl reload nginx'
systemctl list-timers certbot.timer --no-pager | head -3"
}

step_verify() {
  say "local: check the live site"
  local fail=0
  check() {  # url expected-status [header-regex]
    local out code
    out=$(curl -sS -o /dev/null -D - --max-time 20 "$1" 2>&1) || { echo "  FAIL $1: $out"; fail=1; return; }
    code=$(printf '%s' "$out" | head -1 | awk '{print $2}')
    if [ "$code" != "$2" ]; then echo "  FAIL $1: HTTP $code, expected $2"; fail=1; return; fi
    if [ -n "${3:-}" ] && ! printf '%s' "$out" | grep -qiE "$3"; then echo "  FAIL $1: no header /$3/"; fail=1; return; fi
    echo "  ok   $1 -> $code ${3:+($3)}"
  }
  check "http://$DOMAIN/" 301 "^location: https://$DOMAIN/"
  check "https://www.$DOMAIN/" 301 "^location: https://$DOMAIN/"
  check "https://$DOMAIN/" 200 "^content-type: text/html"
  check "https://$DOMAIN/debug.html" 200 "^content-type: text/html"
  check "https://$DOMAIN/model/model.onnx" 200
  check "https://$DOMAIN/model/labels.json" 200 "^content-type: application/json"
  local wasm; wasm=$(cd "$DIST" && ls assets/*.wasm | head -1)
  check "https://$DOMAIN/$wasm" 200 "^content-type: application/wasm"
  check "https://$DOMAIN/mediapipe/wasm/vision_wasm_internal.wasm" 200 "^content-type: application/wasm"
  [ "$fail" -eq 0 ] && echo "Live: https://$DOMAIN/  (debug page: https://$DOMAIN/debug.html)" || exit 1
}

for s in "${STEPS[@]}"; do
  case "$s" in
    dns|build|setup|upload|nginx|cert|verify) "step_$s" || { echo "stopped at step: $s"; exit 1; } ;;
    *) echo "unknown step: $s (dns build setup upload nginx cert verify)"; exit 1 ;;
  esac
done
