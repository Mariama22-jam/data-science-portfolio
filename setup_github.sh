#!/usr/bin/env bash
# Create the GitHub repository and push this folder in one go.
#
# Requirements: git, and the GitHub CLI (https://cli.github.com) logged in with
#   gh auth login
#
# Usage:  ./setup_github.sh            # creates <your-username>/data-science-portfolio (public)
#         ./setup_github.sh my-name    # custom repository name
set -euo pipefail

REPO="${1:-data-science-portfolio}"

command -v gh >/dev/null || { echo "Install the GitHub CLI first: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run 'gh auth login' first."; exit 1; }
USER_NAME="$(gh api user --jq .login)"

cd "$(dirname "$0")"
sed -i.bak "s#YOUR-GITHUB-USERNAME#${USER_NAME}#g; s#/phd-portfolio/#/${REPO}/#g" README.md && rm -f README.md.bak

if [ ! -d .git ]; then
  git init -b main
  git add -A
  git commit -m "Initial portfolio: survival ML on PBC, polymer Monte Carlo, heart-failure ML"
fi

gh repo create "${USER_NAME}/${REPO}" --public --source . --remote origin --push \
  --description "Reproducible statistics, ML and Monte Carlo projects with automated CI"

echo
echo "Done: https://github.com/${USER_NAME}/${REPO}"
echo "CI runs on every push; the 'Full run' workflow regenerates results/ and commits them."
echo "You can also trigger it from the Actions tab (Full run -> Run workflow)."
