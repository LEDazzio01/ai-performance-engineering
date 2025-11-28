#!/usr/bin/env python3
"""
Check for upstream updates to CUTLASS and TransformerEngine.

Source of truth: setup.sh holds all pinned CUTLASS/TE versions (no separate
dependency_versions.json). This script compares those pins against the latest
releases on GitHub and reports if updates are available.

Usage:
    python scripts/check_upstream_versions.py
    python scripts/check_upstream_versions.py --verbose
    python scripts/check_upstream_versions.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


@dataclass
class VersionInfo:
    """Information about a dependency version."""
    name: str
    current_commit: str
    current_version: str
    latest_commit: Optional[str] = None
    latest_version: Optional[str] = None
    latest_tag: Optional[str] = None
    latest_date: Optional[str] = None
    commits_behind: Optional[int] = None
    update_available: bool = False
    release_notes_url: Optional[str] = None
    error: Optional[str] = None


def github_api_request(url: str, timeout: int = 30) -> dict | list | None:
    """Make a GitHub API request."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "upstream-version-checker/1.0",
    }
    
    # Use GitHub token if available (higher rate limits)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        if e.code == 403:
            print(f"⚠️  GitHub API rate limit exceeded. Set GITHUB_TOKEN for higher limits.", file=sys.stderr)
        elif e.code == 404:
            return None
        raise
    except URLError as e:
        print(f"⚠️  Network error: {e}", file=sys.stderr)
        return None


def get_current_versions(project_root: Path) -> dict[str, dict]:
    """Extract current pinned versions from setup.sh."""
    setup_sh = project_root / "setup.sh"
    versions = {
        "cutlass": {"commit": None, "version": None, "repo": "NVIDIA/cutlass"},
        "transformer_engine": {"commit": None, "version": None, "repo": "NVIDIA/TransformerEngine"},
    }
    
    if not setup_sh.exists():
        return versions
    
    content = setup_sh.read_text()
    
    # CUTLASS
    if match := re.search(r'CUTLASS_REF=.*?([a-f0-9]{40})', content):
        versions["cutlass"]["commit"] = match.group(1)
    if match := re.search(r'CUTLASS_TARGET_VERSION.*?(\d+\.\d+\.\d+)', content):
        versions["cutlass"]["version"] = match.group(1)
    
    # TransformerEngine
    if match := re.search(r'TE_GIT_COMMIT=.*?([a-f0-9]{40})', content):
        versions["transformer_engine"]["commit"] = match.group(1)
    
    # Also check version.h for CUTLASS if available
    cutlass_version_h = project_root / "third_party" / "cutlass" / "include" / "cutlass" / "version.h"
    if cutlass_version_h.exists():
        vh_content = cutlass_version_h.read_text()
        major = minor = patch = 0
        for line in vh_content.splitlines():
            if m := re.match(r'#define\s+CUTLASS_MAJOR\s+(\d+)', line):
                major = int(m.group(1))
            elif m := re.match(r'#define\s+CUTLASS_MINOR\s+(\d+)', line):
                minor = int(m.group(1))
            elif m := re.match(r'#define\s+CUTLASS_PATCH\s+(\d+)', line):
                patch = int(m.group(1))
        versions["cutlass"]["version"] = f"{major}.{minor}.{patch}"
    
    return versions


def check_cutlass_updates(current: dict, verbose: bool = False) -> VersionInfo:
    """Check CUTLASS for updates."""
    info = VersionInfo(
        name="CUTLASS",
        current_commit=current.get("commit", "unknown"),
        current_version=current.get("version", "unknown"),
    )
    
    repo = current.get("repo", "NVIDIA/cutlass")
    
    # Get latest release
    releases = github_api_request(f"https://api.github.com/repos/{repo}/releases")
    if releases and len(releases) > 0:
        latest_release = releases[0]
        info.latest_tag = latest_release.get("tag_name", "")
        info.latest_version = info.latest_tag.lstrip("v")
        info.latest_date = latest_release.get("published_at", "")[:10]
        info.release_notes_url = latest_release.get("html_url")
        
        if verbose:
            print(f"  Latest release: {info.latest_tag} ({info.latest_date})")
    
    # Get latest commit on main branch
    commits = github_api_request(f"https://api.github.com/repos/{repo}/commits?per_page=1")
    if commits and len(commits) > 0:
        info.latest_commit = commits[0].get("sha", "")[:12]
        commit_date = commits[0].get("commit", {}).get("committer", {}).get("date", "")
        if verbose:
            print(f"  Latest commit: {info.latest_commit} ({commit_date[:10]})")
    
    # Compare commits to see how far behind we are
    if info.current_commit and info.latest_commit:
        compare = github_api_request(
            f"https://api.github.com/repos/{repo}/compare/{info.current_commit[:12]}...main"
        )
        if compare:
            info.commits_behind = compare.get("ahead_by", 0)
    
    # Determine if update is available
    if info.latest_version and info.current_version:
        def parse_version(v: str) -> tuple:
            return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
        
        try:
            current_v = parse_version(info.current_version)
            latest_v = parse_version(info.latest_version)
            info.update_available = latest_v > current_v
        except (ValueError, IndexError):
            pass
    
    # Also check commits behind
    if info.commits_behind and info.commits_behind > 0:
        info.update_available = True
    
    return info


def check_transformer_engine_updates(current: dict, verbose: bool = False) -> VersionInfo:
    """Check TransformerEngine for updates."""
    info = VersionInfo(
        name="TransformerEngine",
        current_commit=current.get("commit", "unknown"),
        current_version=current.get("version", "unknown"),
    )
    
    repo = current.get("repo", "NVIDIA/TransformerEngine")
    
    # Get latest release
    releases = github_api_request(f"https://api.github.com/repos/{repo}/releases")
    latest_release_commit = None
    if releases and len(releases) > 0:
        latest_release = releases[0]
        info.latest_tag = latest_release.get("tag_name", "")
        info.latest_version = info.latest_tag.lstrip("v")
        info.latest_date = latest_release.get("published_at", "")[:10]
        info.release_notes_url = latest_release.get("html_url")
        
        # Get the commit SHA for the latest release tag
        tag_ref = github_api_request(f"https://api.github.com/repos/{repo}/git/refs/tags/{info.latest_tag}")
        if tag_ref:
            latest_release_commit = tag_ref.get("object", {}).get("sha", "")[:12]
        
        if verbose:
            print(f"  Latest release: {info.latest_tag} ({info.latest_date})")
            if latest_release_commit:
                print(f"  Latest release commit: {latest_release_commit}")
    
    # Get latest commit on main branch
    commits = github_api_request(f"https://api.github.com/repos/{repo}/commits?per_page=1")
    if commits and len(commits) > 0:
        info.latest_commit = commits[0].get("sha", "")[:12]
        commit_date = commits[0].get("commit", {}).get("committer", {}).get("date", "")
        if verbose:
            print(f"  Latest HEAD commit: {info.latest_commit} ({commit_date[:10]})")
    
    # Compare commits to main
    if info.current_commit and len(info.current_commit) >= 12:
        compare = github_api_request(
            f"https://api.github.com/repos/{repo}/compare/{info.current_commit[:12]}...main"
        )
        if compare:
            info.commits_behind = compare.get("ahead_by", 0)
    
    # Check for TE's bundled CUTLASS version in latest
    if verbose:
        gitmodules = github_api_request(
            f"https://api.github.com/repos/{repo}/contents/.gitmodules"
        )
        if gitmodules:
            import base64
            content = base64.b64decode(gitmodules.get("content", "")).decode()
            if "cutlass" in content.lower():
                print(f"  Note: TE bundles CUTLASS as submodule (check .gitmodules for version)")
    
    # Determine if update is available
    # Compare against RELEASE version, not HEAD (we want stable releases)
    if latest_release_commit and info.current_commit:
        current_short = info.current_commit[:12]
        release_short = latest_release_commit[:12]
        
        if current_short == release_short:
            # We're on the latest release
            info.update_available = False
        else:
            # Check if we're behind the latest release
            compare_to_release = github_api_request(
                f"https://api.github.com/repos/{repo}/compare/{current_short}...{info.latest_tag}"
            )
            if compare_to_release:
                behind_release = compare_to_release.get("ahead_by", 0)
                info.update_available = behind_release > 0
            else:
                # Fallback: if current is the release commit, we're up to date
                info.update_available = current_short != release_short
    
    return info


def check_te_bundled_cutlass(verbose: bool = False) -> Optional[str]:
    """Check what CUTLASS version TransformerEngine's latest main bundles."""
    # Get .gitmodules from TE main branch
    gitmodules = github_api_request(
        "https://api.github.com/repos/NVIDIA/TransformerEngine/contents/.gitmodules?ref=main"
    )
    if not gitmodules:
        return None
    
    # Get the submodule commit
    tree = github_api_request(
        "https://api.github.com/repos/NVIDIA/TransformerEngine/git/trees/main"
    )
    if not tree:
        return None
    
    # Find 3rdparty in tree
    for item in tree.get("tree", []):
        if item.get("path") == "3rdparty":
            subtree = github_api_request(item.get("url"))
            if subtree:
                for subitem in subtree.get("tree", []):
                    if subitem.get("path") == "cutlass":
                        cutlass_sha = subitem.get("sha", "")[:12]
                        if verbose:
                            print(f"  TE main bundles CUTLASS commit: {cutlass_sha}")
                        
                        # Try to get version from that commit
                        version_h = github_api_request(
                            f"https://api.github.com/repos/NVIDIA/cutlass/contents/include/cutlass/version.h?ref={cutlass_sha}"
                        )
                        if version_h:
                            import base64
                            content = base64.b64decode(version_h.get("content", "")).decode()
                            major = minor = patch = 0
                            for line in content.splitlines():
                                if m := re.match(r'#define\s+CUTLASS_MAJOR\s+(\d+)', line):
                                    major = int(m.group(1))
                                elif m := re.match(r'#define\s+CUTLASS_MINOR\s+(\d+)', line):
                                    minor = int(m.group(1))
                                elif m := re.match(r'#define\s+CUTLASS_PATCH\s+(\d+)', line):
                                    patch = int(m.group(1))
                            return f"{major}.{minor}.{patch}"
    
    return None


def print_report(cutlass: VersionInfo, te: VersionInfo, te_bundled_cutlass: Optional[str]):
    """Print human-readable report."""
    print()
    print("=" * 70)
    print("Upstream Version Check Report")
    print("=" * 70)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # CUTLASS
    print("CUTLASS")
    print("-" * 40)
    print(f"  Current:  {cutlass.current_version} (commit {cutlass.current_commit[:12] if cutlass.current_commit else 'unknown'})")
    if cutlass.latest_version:
        print(f"  Latest:   {cutlass.latest_version} (tag {cutlass.latest_tag})")
    if cutlass.latest_commit:
        print(f"  Head:     commit {cutlass.latest_commit}")
    if cutlass.commits_behind:
        print(f"  Behind:   {cutlass.commits_behind} commits")
    
    if cutlass.update_available:
        print(f"  ⚠️  UPDATE AVAILABLE")
        if cutlass.release_notes_url:
            print(f"      Release notes: {cutlass.release_notes_url}")
    else:
        print(f"  ✓ Up to date")
    print()
    
    # TransformerEngine
    print("TransformerEngine")
    print("-" * 40)
    print(f"  Current:  commit {te.current_commit[:12] if te.current_commit else 'unknown'}")
    if te.latest_tag:
        print(f"  Latest:   {te.latest_version} (tag {te.latest_tag}, {te.latest_date})")
    if te.latest_commit:
        print(f"  Head:     commit {te.latest_commit}")
    if te.commits_behind:
        print(f"  Behind:   {te.commits_behind} commits")
    
    if te.update_available:
        print(f"  ⚠️  UPDATE AVAILABLE")
        if te.release_notes_url:
            print(f"      Release notes: {te.release_notes_url}")
    else:
        print(f"  ✓ Up to date (or close)")
    print()
    
    # TE's bundled CUTLASS
    if te_bundled_cutlass:
        print("TransformerEngine Bundled CUTLASS")
        print("-" * 40)
        print(f"  TE main bundles: CUTLASS {te_bundled_cutlass}")
        print(f"  Your standalone: CUTLASS {cutlass.current_version}")
        
        def parse_v(v):
            return tuple(int(x) for x in v.split(".")[:3])
        
        try:
            bundled = parse_v(te_bundled_cutlass)
            standalone = parse_v(cutlass.current_version)
            
            if bundled > standalone:
                print(f"  ⚠️  TE's bundled CUTLASS is NEWER than your standalone!")
                print(f"      Consider updating your standalone CUTLASS to {te_bundled_cutlass}")
            elif bundled < standalone:
                print(f"  ✓ Your standalone CUTLASS is newer (symlink workaround still needed)")
            else:
                print(f"  ✓ Versions match - symlink may no longer be needed!")
        except (ValueError, IndexError):
            pass
        print()
    
    # Summary
    print("=" * 70)
    updates_available = cutlass.update_available or te.update_available
    if updates_available:
        print("⚠️  Updates available! Review before upgrading:")
        print("   - Test builds after updating")
        print("   - Check if TE's bundled CUTLASS has SM100a support")
        print("   - Re-run: make verify-deps")
    else:
        print("✓ All dependencies are up to date")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Check for upstream updates to CUTLASS and TransformerEngine"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check-te-cutlass", action="store_true", 
                       help="Also check what CUTLASS version TE's main branch bundles")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    
    if not args.json:
        print("🔍 Checking upstream versions...")
        print()
    
    # Get current versions
    current = get_current_versions(project_root)
    
    if args.verbose and not args.json:
        print("Current pinned versions:")
        print(f"  CUTLASS: {current['cutlass'].get('version', 'unknown')} "
              f"(commit {current['cutlass'].get('commit', 'unknown')[:12]})")
        print(f"  TransformerEngine: commit {current['transformer_engine'].get('commit', 'unknown')[:12]}")
        print()
    
    # Check for updates
    if not args.json:
        print("Checking CUTLASS...")
    cutlass_info = check_cutlass_updates(current["cutlass"], verbose=args.verbose)
    
    if not args.json:
        print("Checking TransformerEngine...")
    te_info = check_transformer_engine_updates(current["transformer_engine"], verbose=args.verbose)
    
    # Check TE's bundled CUTLASS
    te_bundled_cutlass = None
    if args.check_te_cutlass or args.verbose:
        if not args.json:
            print("Checking TE's bundled CUTLASS version...")
        te_bundled_cutlass = check_te_bundled_cutlass(verbose=args.verbose)
    
    if args.json:
        output = {
            "checked_at": datetime.now().isoformat(),
            "cutlass": {
                "current_version": cutlass_info.current_version,
                "current_commit": cutlass_info.current_commit,
                "latest_version": cutlass_info.latest_version,
                "latest_tag": cutlass_info.latest_tag,
                "latest_commit": cutlass_info.latest_commit,
                "commits_behind": cutlass_info.commits_behind,
                "update_available": cutlass_info.update_available,
                "release_notes_url": cutlass_info.release_notes_url,
            },
            "transformer_engine": {
                "current_commit": te_info.current_commit,
                "latest_version": te_info.latest_version,
                "latest_tag": te_info.latest_tag,
                "latest_commit": te_info.latest_commit,
                "commits_behind": te_info.commits_behind,
                "update_available": te_info.update_available,
                "release_notes_url": te_info.release_notes_url,
            },
            "te_bundled_cutlass": te_bundled_cutlass,
            "any_updates_available": cutlass_info.update_available or te_info.update_available,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(cutlass_info, te_info, te_bundled_cutlass)
    
    # Exit code: 0 if up to date, 1 if updates available
    return 1 if (cutlass_info.update_available or te_info.update_available) else 0


if __name__ == "__main__":
    sys.exit(main())
