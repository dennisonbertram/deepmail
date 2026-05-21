"""Unified CLI entry point for Deep Email.

Dispatches subcommands:
  deep-email              → starts MCP server (stdio transport)
  deep-email serve        → starts MCP server (explicit)
  deep-email auth         → OAuth flow
  deep-email whoami       → check auth status
  deep-email query QUERY  → search Gmail
  deep-email run SEED     → run the build pipeline
  deep-email init         → write .mcp.json
  deep-email setup        → interactive wizard
  deep-email --help       → shows all commands
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click


def _load_dotenv_quietly() -> None:
    """Best-effort .env loader."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


# ---- CLI group ----


@click.group(
    invoke_without_command=True,
    help=(
        "Deep Email: deep email understanding for AI agents.\n\n"
        "Run with no subcommand to start the MCP server (stdio transport).\n"
        "Use 'deep-email setup' for first-time configuration."
    ),
)
@click.version_option("0.3.0", prog_name="deep-email")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Top-level CLI group. No subcommand → MCP server."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _load_dotenv_quietly()

    if ctx.invoked_subcommand is None:
        # No subcommand → start MCP server
        _start_mcp_server()


# ---- serve ----


@cli.command()
def serve() -> None:
    """Start the MCP server (stdio transport)."""
    _start_mcp_server()


def _start_mcp_server() -> None:
    """Import and run the MCP server."""
    from pi_email.mcp_server import main as mcp_main

    mcp_main()


# ---- auth ----


@cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="Re-run the consent flow even if a token already exists.",
)
@click.option(
    "--refresh-auth",
    "refresh_auth",
    is_flag=True,
    help="Force re-consent to pick up new OAuth scopes.",
)
@click.option(
    "--remove",
    "remove_account",
    default=None,
    help="Remove an account's credentials (pass email address).",
)
@click.pass_context
def auth(ctx: click.Context, force: bool, refresh_auth: bool, remove_account: str | None) -> None:
    """Authenticate a Google account (can be run multiple times for multi-account)."""
    from pi_email.token_store import TokenStore

    if remove_account:
        store = TokenStore()
        if store.remove(remove_account):
            click.echo(f"Removed credentials for {remove_account}")
        else:
            click.echo(f"No credentials found for {remove_account}")
        return

    from pi_email.oauth import OAuthConfig, acquire_credentials, refresh_if_needed

    try:
        config = OAuthConfig.from_env()
    except RuntimeError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(2)

    store = TokenStore()

    if not force and not refresh_auth:
        # In multi-account mode, we always allow adding new accounts.
        # Only skip the flow if we detect the SAME account trying to re-auth.
        pass  # Fall through to the consent flow.

    click.echo("Opening browser for Google consent...", err=True)
    try:
        creds = acquire_credentials(config, open_browser=True)
    except Exception as exc:
        click.echo(f"OAuth flow failed: {exc}", err=True)
        raise SystemExit(2)

    # Determine the email for this account.
    email = _resolve_email_from_creds(creds)
    if not email:
        click.echo("Could not determine email address from credentials.", err=True)
        raise SystemExit(2)

    try:
        store.save(creds, email=email)
    except Exception as exc:
        click.echo(f"Could not persist token: {exc}", err=True)
        raise SystemExit(2)

    click.echo(f"Authenticated as {email}")


def _resolve_email_from_creds(creds) -> str | None:
    """Resolve the email address for freshly acquired credentials."""
    try:
        from googleapiclient.discovery import build
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = service.users().getProfile(userId="me").execute()
        return profile.get("emailAddress")
    except Exception:
        return None


# ---- accounts ----


@cli.command()
def accounts() -> None:
    """List all authenticated Gmail accounts."""
    from pi_email.token_store import TokenStore
    from pi_email.oauth import refresh_if_needed

    store = TokenStore()
    account_list = store.list_accounts()

    if not account_list:
        click.echo("No authenticated accounts. Run: deep-email auth")
        raise SystemExit(1)

    click.echo("Authenticated accounts:")
    for i, email in enumerate(account_list, 1):
        creds = store.load(email)
        status = ""
        if creds:
            try:
                refresh_if_needed(creds)
                # Try to get message count.
                try:
                    from googleapiclient.discovery import build
                    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
                    profile = service.users().getProfile(userId="me").execute()
                    msg_count = profile.get("messagesTotal", 0)
                    status = f" ({msg_count:,} messages)"
                except Exception:
                    status = ""
            except Exception:
                status = " (token expired)"
        click.echo(f"  {i}. {email}{status}")


# ---- whoami ----


@cli.command()
def whoami() -> None:
    """Check authentication status and account info."""
    from pi_email.token_store import TokenStore
    from pi_email.oauth import refresh_if_needed

    store = TokenStore()
    all_accounts = store.load_all()

    if not all_accounts:
        click.echo("Not authenticated. Run: deep-email auth")
        raise SystemExit(1)

    from googleapiclient.discovery import build

    for email, creds in all_accounts.items():
        try:
            creds = refresh_if_needed(creds)
        except Exception as e:
            click.echo(f"[{email}] Token expired or invalid: {e}")
            continue

        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        profile = service.users().getProfile(userId="me").execute()

        click.echo(f"email:           {profile.get('emailAddress', 'unknown')}")
        click.echo(f"messages_total:  {profile.get('messagesTotal', 'unknown')}")
        click.echo(f"threads_total:   {profile.get('threadsTotal', 'unknown')}")
        if len(all_accounts) > 1:
            click.echo("---")


# ---- query ----


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--max",
    "max_results",
    type=int,
    default=50,
    show_default=True,
    help="Cap on messages fetched.",
)
@click.pass_context
def query(ctx: click.Context, query: tuple[str, ...], max_results: int) -> None:
    """Run a Gmail search and print matches."""
    from pi_email.cli import query as _query_cmd, main as _cli_main

    sub_ctx = click.Context(_cli_main, info_name="deep-email")
    sub_ctx.ensure_object(dict)
    sub_ctx.obj["verbose"] = ctx.obj.get("verbose", False)
    with sub_ctx:
        sub_ctx.invoke(_query_cmd, query=query, max_results=max_results)


# ---- run ----


@cli.command()
@click.argument("seed", required=True)
@click.option(
    "--source",
    type=click.Choice(["fixture", "gmail"], case_sensitive=False),
    default="fixture",
    show_default=True,
    help="Which Searcher to drive the loop against.",
)
@click.option("--max-corpus", type=int, default=100, show_default=True)
@click.option("--max-per-query", type=int, default=500, show_default=True)
@click.option(
    "--profiles-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--fixtures-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--force-mock", is_flag=True)
@click.option("--skip-judge", is_flag=True)
@click.option("--strict-extract/--no-strict-extract", default=None)
@click.pass_context
def run(ctx: click.Context, seed: str, **kwargs) -> None:
    """Drive the expansion loop end-to-end."""
    from pi_email.cli import run as _run_cmd, main as _cli_main

    sub_ctx = click.Context(_cli_main, info_name="deep-email")
    sub_ctx.ensure_object(dict)
    sub_ctx.obj["verbose"] = ctx.obj.get("verbose", False)
    with sub_ctx:
        sub_ctx.invoke(_run_cmd, seed=seed, **kwargs)


# ---- init ----


def _write_mcp_json(mcp_json_path: Path) -> None:
    """Write or merge the deep-email entry into .mcp.json."""
    deepmail_config = {
        "type": "stdio",
        "command": "uvx",
        "args": ["deep-email"],
    }

    if mcp_json_path.exists():
        existing = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        existing.setdefault("mcpServers", {})["deepmail"] = deepmail_config
        config = existing
    else:
        config = {"mcpServers": {"deepmail": deepmail_config}}

    mcp_json_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


@cli.command()
def init() -> None:
    """Configure Deep Email MCP server in the current project.

    Writes a .mcp.json file (or merges into an existing one) so your
    AI agent can discover the Deep Email MCP server.
    """
    mcp_json_path = Path.cwd() / ".mcp.json"

    if mcp_json_path.exists():
        try:
            existing = json.loads(mcp_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        if "deepmail" in existing.get("mcpServers", {}):
            click.echo("Deep Email is already configured in .mcp.json")
            return

    _write_mcp_json(mcp_json_path)
    click.echo(f"Wrote {mcp_json_path}")
    click.echo("  Restart your agent to activate Deep Email.")


# ---- setup ----


@cli.command()
def setup() -> None:
    """Interactive setup wizard -- credentials, auth, and agent config."""
    click.echo("Welcome to Deep Email!\n")

    # Step 1: Credentials
    click.echo("Step 1: Google Cloud credentials")
    custom_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if custom_id:
        click.echo(f"  [ok] Using custom GOOGLE_CLIENT_ID from env (length={len(custom_id)})")
    else:
        click.echo("  [ok] Using built-in Deepmail credentials")
        click.echo("  (Set GOOGLE_CLIENT_ID in env to use your own)")

    # Step 2: Authenticate
    click.echo("\nStep 2: Gmail authentication")
    from pi_email.token_store import TokenStore

    store = TokenStore()
    existing_accounts = store.list_accounts()
    if existing_accounts:
        click.echo(f"  [ok] Already authenticated ({len(existing_accounts)} account(s))")
        for acct in existing_accounts:
            click.echo(f"       - {acct}")
    else:
        if click.confirm("  Open browser for Google consent?", default=True):
            from pi_email.oauth import OAuthConfig, acquire_credentials

            try:
                config = OAuthConfig.from_env()
                creds = acquire_credentials(config)
                email = _resolve_email_from_creds(creds)
                if email:
                    store.save(creds, email=email)
                    click.echo(f"  [ok] Authenticated as {email}")
                else:
                    click.echo("  [!!] Could not determine email address.")
            except Exception as exc:
                click.echo(f"  [!!] Authentication failed: {exc}")
                click.echo("  Run 'deep-email auth' later to retry.")
        else:
            click.echo("  Skipping -- run 'deep-email auth' later")

    # Step 3: Agent config
    click.echo("\nStep 3: Agent configuration")
    mcp_json = Path.cwd() / ".mcp.json"
    if mcp_json.exists() and "deepmail" in mcp_json.read_text(encoding="utf-8"):
        click.echo("  [ok] .mcp.json already configured")
    else:
        if click.confirm("  Write .mcp.json in current directory?", default=True):
            _write_mcp_json(mcp_json)
            click.echo(f"  [ok] Wrote {mcp_json}")
        else:
            click.echo("  Skipping -- run 'deep-email init' later")

    click.echo("\nSetup complete! Restart your agent to start using Deep Email.")


# ---- entry point ----


def main() -> None:
    """Entry point for the `deep-email` console script."""
    cli()


if __name__ == "__main__":
    main()
