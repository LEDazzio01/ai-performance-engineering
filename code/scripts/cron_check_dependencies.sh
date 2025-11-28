#!/usr/bin/env bash
# Cron wrapper for dependency update alerts
#
# This script is designed to be run from cron. It only produces output
# (and sends alerts) when updates are available.
#
# Setup:
#   1. Set environment variables (in crontab or /etc/environment):
#      SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
#      # Optional email settings:
#      ALERT_EMAIL=admin@example.com
#      SMTP_SERVER=smtp.example.com
#
#   2. Add to crontab (check daily at 9am):
#      0 9 * * * /path/to/code/scripts/cron_check_dependencies.sh
#
#   3. Or weekly on Monday:
#      0 9 * * 1 /path/to/code/scripts/cron_check_dependencies.sh
#
# The script will:
#   - Only alert if updates are available
#   - Send to Slack if SLACK_WEBHOOK_URL is set
#   - Send email if ALERT_EMAIL and SMTP_SERVER are set
#   - Write status to /tmp/dependency_status.json for dashboard
#   - Log to /var/log/dependency_check.log (if writable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/dependency_check.log}"
STATUS_FILE="${STATUS_FILE:-/tmp/dependency_status.json}"

cd "${PROJECT_ROOT}"

# Build alert arguments based on environment
ALERT_ARGS="--quiet --status-file ${STATUS_FILE}"

if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    ALERT_ARGS="${ALERT_ARGS} --slack-webhook ${SLACK_WEBHOOK_URL}"
fi

if [ -n "${ALERT_EMAIL:-}" ] && [ -n "${SMTP_SERVER:-}" ]; then
    ALERT_ARGS="${ALERT_ARGS} --email ${ALERT_EMAIL} --smtp-server ${SMTP_SERVER}"
    if [ -n "${SMTP_PORT:-}" ]; then
        ALERT_ARGS="${ALERT_ARGS} --smtp-port ${SMTP_PORT}"
    fi
    if [ -n "${SMTP_USER:-}" ]; then
        ALERT_ARGS="${ALERT_ARGS} --smtp-user ${SMTP_USER}"
    fi
    if [ -n "${SMTP_PASSWORD:-}" ]; then
        ALERT_ARGS="${ALERT_ARGS} --smtp-password ${SMTP_PASSWORD}"
    fi
fi

# Run the alert script
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
EXIT_CODE=0

# shellcheck disable=SC2086
python3 scripts/alert_dependency_updates.py ${ALERT_ARGS} 2>&1 || EXIT_CODE=$?

# Log result if log file is writable
if [ -w "$(dirname "${LOG_FILE}")" ] 2>/dev/null || [ -w "${LOG_FILE}" ] 2>/dev/null; then
    case ${EXIT_CODE} in
        0) echo "${TIMESTAMP} - OK: All dependencies up to date" >> "${LOG_FILE}" ;;
        1) echo "${TIMESTAMP} - ALERT: Updates available, notifications sent" >> "${LOG_FILE}" ;;
        2) echo "${TIMESTAMP} - ERROR: Failed to send some notifications" >> "${LOG_FILE}" ;;
        *) echo "${TIMESTAMP} - ERROR: Unknown exit code ${EXIT_CODE}" >> "${LOG_FILE}" ;;
    esac
fi

exit ${EXIT_CODE}



