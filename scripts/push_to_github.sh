#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
PROXY="${PROXY:-http://127.0.0.1:7890}"

if [[ -n "${PROXY}" ]]; then
  export http_proxy="$PROXY"
  export https_proxy="$PROXY"
  export HTTP_PROXY="$PROXY"
  export HTTPS_PROXY="$PROXY"
fi

echo "Repository: $REPO_ROOT"
echo "Remote: $REMOTE"
echo "Branch: $BRANCH"
echo "Proxy: ${PROXY:-none}"
echo

git status
echo
git remote -v
echo

if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  echo "Branch '$BRANCH' not found." >&2
  exit 1
fi

echo "Fetching remote history..."
git fetch "$REMOTE" "$BRANCH" || true

if git rev-parse --verify "$REMOTE/$BRANCH" >/dev/null 2>&1; then
  echo "Remote branch exists; rebasing local commits..."
  git pull --rebase "$REMOTE" "$BRANCH"
else
  echo "Remote branch not found; creating it on push."
fi

echo "Pushing to GitHub..."
git push -u "$REMOTE" "$BRANCH"

echo "Done."
