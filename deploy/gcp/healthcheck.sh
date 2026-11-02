#!/usr/bin/env bash
# Health checks. On the VM:        ./healthcheck.sh local
#                from anywhere:    SCOUT_GATEWAY_URL=https://<host> ./healthcheck.sh remote
# Secret values are read from /etc/scout (local mode needs root) and never printed.
set -uo pipefail
ok() { printf 'ok    %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; rc=1; }
rc=0
case "${1:-}" in
  local)
    set -a; . /etc/scout/vllm.env; set +a
    systemctl is-active --quiet wayfinder-vllm && ok "wayfinder-vllm active" || bad "wayfinder-vllm not active (journalctl -u wayfinder-vllm -n 50)"
    systemctl is-active --quiet scout-gateway && ok "scout-gateway active" || bad "scout-gateway not active"
    systemctl is-active --quiet caddy && ok "caddy active" || bad "caddy not active"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    curl -fs -m 5 http://127.0.0.1:8000/health >/dev/null && ok "vLLM /health" || bad "vLLM /health (still loading? first start takes several minutes)"
    curl -fs -m 5 -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/v1/models | grep -q wayfinder-glimmer && ok "model listed: wayfinder-glimmer" || bad "model not listed"
    # One real tool-call turn through vLLM (this is what Scout's explore loop sends).
    body='{"model":"wayfinder-glimmer","max_tokens":256,"temperature":0.85,"top_p":0.95,"tool_choice":"auto",
      "tools":[{"type":"function","function":{"name":"grep","description":"Search the repository","parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}],
      "messages":[{"role":"system","content":"You are a code-search subagent. Use the grep tool."},{"role":"user","content":"Where is the HTTP retry backoff configured?"}]}'
    curl -fs -m 120 -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' -d "$body" http://127.0.0.1:8000/v1/chat/completions | grep -q '"tool_calls"' && ok "tool call parsed (muse_glimmer parser)" || bad "no parsed tool call"
    curl -fs -m 5 http://127.0.0.1:8080/v1/healthz >/dev/null && ok "gateway /v1/healthz (localhost)" || bad "gateway /v1/healthz"
    ss -ltnH | awk '{print $4}' | grep -E ':(8000|8080)$' | grep -v -E '^(127\.0\.0\.1|\[::1\]):' && bad "vLLM or gateway listens on a non-loopback address" || ok "vLLM and gateway listen on loopback only"
    ;;
  remote)
    : "${SCOUT_GATEWAY_URL:?set SCOUT_GATEWAY_URL}"
    curl -fs -m 10 "$SCOUT_GATEWAY_URL/v1/healthz" && echo && ok "gateway reachable over HTTPS" || bad "gateway not reachable at $SCOUT_GATEWAY_URL"
    code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$SCOUT_GATEWAY_URL/v1/warm" -X POST)"
    [ "$code" = 401 ] || [ "$code" = 403 ] && ok "unauthenticated /v1/warm refused ($code)" || bad "unauthenticated /v1/warm returned $code"
    host="${SCOUT_GATEWAY_URL#https://}"; host="${host%%/*}"
    curl -s -m 5 -o /dev/null "http://$host:8000/health" && bad "port 8000 (vLLM) is reachable from outside" || ok "port 8000 closed to the internet"
    curl -s -m 5 -o /dev/null "http://$host:8080/v1/healthz" && bad "port 8080 reachable from outside" || ok "port 8080 closed to the internet"
    ;;
  *) echo "usage: $0 local|remote" >&2; exit 2 ;;
esac
exit $rc
