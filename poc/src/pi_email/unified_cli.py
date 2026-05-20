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
@click.version_option("0.1.0", prog_name="deep-email")
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
@click.pass_context
def auth(ctx: click.Context, force: bool, refresh_auth: bool) -> None:
    """Run the Google OAuth flow and persist tokens."""
    from pi_email.cli import auth as _auth_cmd, main as _cli_main

    # Delegate to the existing cli.py auth command via Click's invoke.
    # We build a fresh context from the old CLI group so all the
    # decorators / options resolve correctly.
    runner = click.testing.CliRunner(mix_stderr=False)
    args = ["auth"]
    if force:
        args.append("--force")
    if refresh_auth:
        args.append("--refresh-auth")
    result = runner.invoke(_cli_main, args)
    # Replay output to the real stderr/stdout.
    if result.output:
        click.echo(result.output, nl=False)
    if result.stderr_bytes:
        click.echo(result.stderr_bytes.decode("utf-8", errors="replace"), err=True, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


# ---- whoami ----


@cli.command()
@click.pass_context
def whoami(ctx: click.Context) -> None:
    """Confirm auth by calling Gmail's users.getProfile."""
    from pi_email.cli import main as _cli_main

    runner = click.testing.CliRunner(mix_stderr=False)
    result = runner.invoke(_cli_main, ["whoami"])
    if result.output:
        click.echo(result.output, nl=False)
    if result.stderr_bytes:
        click.echo(result.stderr_bytes.decode("utf-8", errors="replace"), err=True, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


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
    from pi_email.cli import main as _cli_main

    args = ["query"] + list(query) + ["--max", str(max_results)]
    runner = click.testing.CliRunner(mix_stderr=False)
    result = runner.invoke(_cli_main, args)
    if result.output:
        click.echo(result.output, nl=False)
    if result.stderr_bytes:
        click.echo(result.stderr_bytes.decode("utf-8", errors="replace"), err=True, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


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
    from pi_email.cli import main as _cli_main

    args = ["run", seed]
    if kwargs.get("source"):
        args += ["--source", kwargs["source"]]
    if kwargs.get("max_corpus") is not None:
        args += ["--max-corpus", str(kwargs["max_corpus"])]
    if kwargs.get("max_per_query") is not None:
        args += ["--max-per-query", str(kwargs["max_per_query"])]
    if kwargs.get("profiles_dir"):
        args += ["--profiles-dir", str(kwargs["profiles_dir"])]
    if kwargs.get("fixtures_dir"):
        args += ["--fixtures-dir", str(kwargs["fixtures_dir"])]
    if kwargs.get("force_mock"):
        args.append("--force-mock")
    if kwargs.get("skip_judge"):
        args.append("--skip-judge")
    if kwargs.get("strict_extract") is True:
        args.append("--strict-extract")
    elif kwargs.get("strict_extract") is False:
        args.append("--no-strict-extract")

    verbose = ctx.obj.get("verbose", False)
    if verbose:
        args = ["-v"] + args

    runner = click.testing.CliRunner(mix_stderr=False)
    result = runner.invoke(_cli_main, args)
    if result.output:
        click.echo(result.output, nl=False)
    if result.stderr_bytes:
        click.echo(result.stderr_bytes.decode("utf-8", errors="replace"), err=True, nl=False)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


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

    # Step 1: Check GOOGLE_CLIENT_ID
    click.echo("Step 1: Google Cloud credentials")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if client_id:
        click.echo(f"  [ok] GOOGLE_CLIENT_ID found (length={len(client_id)})")
    else:
        click.echo("  [!!] GOOGLE_CLIENT_ID not set")
        click.echo("  You need a Google Cloud OAuth client ID.")
        click.echo("  See: https://github.com/user/deep-email#google-cloud-setup")
        click.echo("")
        client_id = click.prompt(
            "  Paste your Client ID (or press Enter to skip)",
            default="",
            show_default=False,
        )
        if client_id:
            click.echo(f"\n  Add this to your shell config (~/.zshrc):")
            click.echo(f'  export GOOGLE_CLIENT_ID="{client_id}"')
            click.echo(f"\n  Or create a .env file:")
            click.echo(f"  echo 'GOOGLE_CLIENT_ID={client_id}' > .env")
            os.environ["GOOGLE_CLIENT_ID"] = client_id  # set for this session
        else:
            click.echo(
                "  Skipping -- set GOOGLE_CLIENT_ID before running 'deep-email auth'"
            )
            return

    # Step 2: Authenticate
    click.echo("\nStep 2: Gmail authentication")
    from pi_email.token_store import TokenStore

    creds = TokenStore().load()
    if creds and not creds.expired:
        click.echo("  [ok] Already authenticated")
    else:
        if click.confirm("  Open browser for Google consent?", default=True):
            from pi_email.oauth import OAuthConfig, acquire_credentials

            try:
                config = OAuthConfig.from_env()
                creds = acquire_credentials(config)
                TokenStore().save(creds)
                click.echo("  [ok] Authenticated")
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
