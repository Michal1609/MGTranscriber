"""Streamable HTTP endpoint of the MCP server at ``/mcp``, sharing the web server's queue and index.

Stateless mode: every request stands alone (the 2026-07-28 protocol needs no session;
older clients still work, each request is handled on a fresh transport). Host checks
and the API key are enforced by ``security.SecurityMiddleware`` like for every route,
so the SDK's own DNS-rebinding check is switched off.
"""

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings

from .. import netguard
from ..mcp_server import LocalJobs, build_server, web_url_for

MCP_PATH = "/mcp"


def mount_mcp(app: FastAPI) -> None:
    """Add the /mcp route; ``app.state.mcp.session_manager.run()`` must wrap the app's lifetime."""
    state = app.state
    settings = state.settings
    server = build_server(settings, state.library, state.search, LocalJobs(state.queue),
                          allow_local_paths=settings.local_paths_allowed, public_urls_only=netguard.guard_needed(settings),
                          web_url=None if settings.host in ("0.0.0.0", "::", "") else web_url_for(settings))
    http_app = server.streamable_http_app(
        streamable_http_path=MCP_PATH, stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    app.router.routes.extend(http_app.routes)
    state.mcp = server
