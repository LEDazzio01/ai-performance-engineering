#!/usr/bin/env python3
"""
Alert when CUTLASS/TransformerEngine updates are available.

Supports multiple notification channels:
  - Slack webhook
  - Email (SMTP)
  - File output (for cron logs)
  - Dashboard status file
  - Console (default)

Usage:
    # Console only (default)
    python scripts/alert_dependency_updates.py

    # Slack webhook
    python scripts/alert_dependency_updates.py --slack-webhook https://hooks.slack.com/services/XXX

    # Email
    python scripts/alert_dependency_updates.py --email admin@example.com --smtp-server smtp.example.com

    # Multiple channels
    python scripts/alert_dependency_updates.py --slack-webhook $SLACK_URL --email admin@example.com

    # Write status to file (for dashboard)
    python scripts/alert_dependency_updates.py --status-file /tmp/dep_status.json

    # Quiet mode (only alert if updates available)
    python scripts/alert_dependency_updates.py --quiet --slack-webhook $SLACK_URL

Environment variables:
    SLACK_WEBHOOK_URL - Default Slack webhook URL
    ALERT_EMAIL - Default email recipient
    SMTP_SERVER - Default SMTP server
    SMTP_PORT - SMTP port (default: 587)
    SMTP_USER - SMTP username (optional)
    SMTP_PASSWORD - SMTP password (optional)

Cron example (check daily at 9am):
    0 9 * * * cd /path/to/code && python scripts/alert_dependency_updates.py --quiet --slack-webhook $SLACK_URL
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


@dataclass
class UpdateStatus:
    """Status of dependency updates."""
    checked_at: str
    cutlass_update: bool
    cutlass_current: str
    cutlass_latest: str
    cutlass_commits_behind: int
    te_update: bool
    te_current: str
    te_latest: str
    te_commits_behind: int
    te_bundled_cutlass: str
    symlink_still_needed: bool
    any_updates: bool
    message: str


def check_updates() -> UpdateStatus:
    """Run the version checker and return status."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Import the checker
    sys.path.insert(0, str(script_dir))
    from check_upstream_versions import (
        get_current_versions,
        check_cutlass_updates,
        check_transformer_engine_updates,
        check_te_bundled_cutlass,
    )
    
    current = get_current_versions(project_root)
    cutlass_info = check_cutlass_updates(current["cutlass"])
    te_info = check_transformer_engine_updates(current["transformer_engine"])
    te_bundled = check_te_bundled_cutlass() or "unknown"
    
    # Determine if symlink is still needed
    def parse_version(v: str) -> tuple:
        try:
            return tuple(int(x) for x in v.split(".")[:3])
        except (ValueError, AttributeError):
            return (0, 0, 0)
    
    bundled_v = parse_version(te_bundled)
    symlink_needed = bundled_v < (4, 3, 0)
    
    any_updates = cutlass_info.update_available or te_info.update_available
    
    # Build message
    messages = []
    if cutlass_info.update_available:
        messages.append(f"CUTLASS: {cutlass_info.current_version} → {cutlass_info.latest_version}")
    if te_info.update_available:
        messages.append(f"TransformerEngine: {te_info.current_commit[:8]} → {te_info.latest_tag}")
    
    if not symlink_needed:
        messages.append("🎉 TE now bundles CUTLASS 4.3.0+ - symlink workaround may be removable!")
    
    message = "; ".join(messages) if messages else "All dependencies up to date"
    
    return UpdateStatus(
        checked_at=datetime.now().isoformat(),
        cutlass_update=cutlass_info.update_available,
        cutlass_current=cutlass_info.current_version or "unknown",
        cutlass_latest=cutlass_info.latest_version or "unknown",
        cutlass_commits_behind=cutlass_info.commits_behind or 0,
        te_update=te_info.update_available,
        te_current=te_info.current_commit[:12] if te_info.current_commit else "unknown",
        te_latest=te_info.latest_tag or "unknown",
        te_commits_behind=te_info.commits_behind or 0,
        te_bundled_cutlass=te_bundled,
        symlink_still_needed=symlink_needed,
        any_updates=any_updates,
        message=message,
    )


def send_slack_alert(webhook_url: str, status: UpdateStatus) -> bool:
    """Send alert to Slack webhook."""
    if not status.any_updates and status.symlink_still_needed:
        return True  # Nothing to alert
    
    # Build Slack message
    if status.any_updates:
        color = "warning"  # yellow
        title = "🔔 Dependency Updates Available"
    elif not status.symlink_still_needed:
        color = "good"  # green
        title = "🎉 CUTLASS Symlink May Be Removable"
    else:
        return True  # Nothing interesting
    
    fields = []
    
    if status.cutlass_update:
        fields.append({
            "title": "CUTLASS",
            "value": f"`{status.cutlass_current}` → `{status.cutlass_latest}` ({status.cutlass_commits_behind} commits)",
            "short": True
        })
    
    if status.te_update:
        fields.append({
            "title": "TransformerEngine", 
            "value": f"`{status.te_current}` → `{status.te_latest}` ({status.te_commits_behind} commits behind HEAD)",
            "short": True
        })
    
    fields.append({
        "title": "TE Bundled CUTLASS",
        "value": f"`{status.te_bundled_cutlass}` (symlink {'still needed' if status.symlink_still_needed else '**may be removable**'})",
        "short": True
    })
    
    payload = {
        "attachments": [{
            "color": color,
            "title": title,
            "text": status.message,
            "fields": fields,
            "footer": "AI Performance Engineering | Dependency Monitor",
            "ts": int(datetime.now().timestamp())
        }]
    }
    
    try:
        data = json.dumps(payload).encode()
        req = Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            return response.status == 200
    except Exception as e:
        print(f"Slack alert failed: {e}", file=sys.stderr)
        return False


def send_email_alert(
    recipient: str,
    status: UpdateStatus,
    smtp_server: str,
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    use_tls: bool = True,
) -> bool:
    """Send alert via email."""
    if not status.any_updates and status.symlink_still_needed:
        return True  # Nothing to alert
    
    # Build email
    if status.any_updates:
        subject = "🔔 CUTLASS/TE Dependency Updates Available"
    else:
        subject = "🎉 CUTLASS Symlink May Be Removable"
    
    body = f"""
AI Performance Engineering - Dependency Update Alert
=====================================================

Checked at: {status.checked_at}

Summary: {status.message}

Details:
--------
CUTLASS:
  Current: {status.cutlass_current}
  Latest:  {status.cutlass_latest}
  Update:  {'Yes' if status.cutlass_update else 'No'}
  Behind:  {status.cutlass_commits_behind} commits

TransformerEngine:
  Current: {status.te_current}
  Latest:  {status.te_latest}
  Update:  {'Yes' if status.te_update else 'No'}
  Behind:  {status.te_commits_behind} commits (from HEAD)

TE Bundled CUTLASS: {status.te_bundled_cutlass}
Symlink Workaround: {'Still needed' if status.symlink_still_needed else 'MAY BE REMOVABLE!'}

Actions:
--------
1. Review changes: make check-updates
2. Update if needed: Edit setup.sh version pins
3. Verify after update: make verify-deps

--
AI Performance Engineering Dependency Monitor
"""
    
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = smtp_user or "noreply@localhost"
    msg["To"] = recipient
    msg.attach(MIMEText(body, "plain"))
    
    try:
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], [recipient], msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], [recipient], msg.as_string())
        return True
    except Exception as e:
        print(f"Email alert failed: {e}", file=sys.stderr)
        return False


def write_status_file(path: str, status: UpdateStatus) -> bool:
    """Write status to JSON file (for dashboard integration)."""
    try:
        data = asdict(status)
        Path(path).write_text(json.dumps(data, indent=2))
        return True
    except Exception as e:
        print(f"Status file write failed: {e}", file=sys.stderr)
        return False


def print_console_alert(status: UpdateStatus, quiet: bool = False) -> None:
    """Print alert to console."""
    if quiet and not status.any_updates and status.symlink_still_needed:
        return
    
    print()
    print("=" * 60)
    if status.any_updates:
        print("🔔 DEPENDENCY UPDATES AVAILABLE")
    elif not status.symlink_still_needed:
        print("🎉 CUTLASS SYMLINK MAY BE REMOVABLE")
    else:
        print("✓ ALL DEPENDENCIES UP TO DATE")
    print("=" * 60)
    print(f"Checked: {status.checked_at}")
    print()
    print(f"CUTLASS:           {status.cutlass_current} → {status.cutlass_latest}" + 
          (" ⚠️ UPDATE" if status.cutlass_update else " ✓"))
    print(f"TransformerEngine: {status.te_current} → {status.te_latest}" +
          (" ⚠️ UPDATE" if status.te_update else " ✓"))
    print(f"TE Bundled CUTLASS: {status.te_bundled_cutlass}")
    print(f"Symlink Needed:    {'Yes' if status.symlink_still_needed else 'NO - may be removable!'}")
    print()
    print(f"Message: {status.message}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Alert when CUTLASS/TransformerEngine updates are available"
    )
    parser.add_argument(
        "--slack-webhook",
        default=os.environ.get("SLACK_WEBHOOK_URL"),
        help="Slack webhook URL"
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("ALERT_EMAIL"),
        help="Email recipient"
    )
    parser.add_argument(
        "--smtp-server",
        default=os.environ.get("SMTP_SERVER"),
        help="SMTP server hostname"
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.environ.get("SMTP_PORT", "587")),
        help="SMTP port (default: 587)"
    )
    parser.add_argument(
        "--smtp-user",
        default=os.environ.get("SMTP_USER"),
        help="SMTP username"
    )
    parser.add_argument(
        "--smtp-password",
        default=os.environ.get("SMTP_PASSWORD"),
        help="SMTP password"
    )
    parser.add_argument(
        "--status-file",
        help="Write status to JSON file (for dashboard)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output if updates are available"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON to stdout"
    )
    
    args = parser.parse_args()
    
    # Check for updates
    if not args.quiet:
        print("🔍 Checking for dependency updates...", file=sys.stderr)
    
    try:
        status = check_updates()
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "403" in error_msg:
            print("⚠️  GitHub API rate limit exceeded.", file=sys.stderr)
            print("   Set GITHUB_TOKEN environment variable for higher limits.", file=sys.stderr)
            print("   Get a token at: https://github.com/settings/tokens", file=sys.stderr)
        else:
            print(f"Error checking updates: {e}", file=sys.stderr)
        return 3  # Distinct exit code for API errors
    
    success = True
    
    # Send alerts
    if args.slack_webhook:
        if not args.quiet:
            print("📤 Sending Slack alert...", file=sys.stderr)
        if not send_slack_alert(args.slack_webhook, status):
            success = False
    
    if args.email and args.smtp_server:
        if not args.quiet:
            print("📧 Sending email alert...", file=sys.stderr)
        if not send_email_alert(
            args.email,
            status,
            args.smtp_server,
            args.smtp_port,
            args.smtp_user,
            args.smtp_password,
        ):
            success = False
    
    if args.status_file:
        if not args.quiet:
            print(f"💾 Writing status to {args.status_file}...", file=sys.stderr)
        if not write_status_file(args.status_file, status):
            success = False
    
    # Output
    if args.json:
        print(json.dumps(asdict(status), indent=2))
    else:
        print_console_alert(status, quiet=args.quiet)
    
    # Exit code: 0 if no updates, 1 if updates available, 2 if alert failed
    if not success:
        return 2
    return 1 if status.any_updates else 0


if __name__ == "__main__":
    sys.exit(main())

