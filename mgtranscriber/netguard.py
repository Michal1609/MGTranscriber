"""Server-side request forgery guard: refuse URLs that point at this computer or the local network.

A server reachable from the network fetches URLs its clients give it (yt-dlp downloads, feeds). Without
this check a client could make it call services that only the server can reach (router admin pages,
other machines on the LAN, cloud metadata endpoints). On loopback the client is the user on this
computer anyway, so the guard is only needed when the server listens on another address.

All addresses a name resolves to must be public. The check runs before the request; a name that
changes its answer between the check and the request (DNS rebinding) and redirects followed by
yt-dlp are not covered, which is why an API key is recommended as well.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


class BlockedAddress(ValueError):
    def __init__(self, host: str):
        super().__init__(f"{host} is on this computer or the local network. Only public addresses are allowed "
                         "while the server is reachable from the network.")
        self.host = host


_EMBEDDING = (  # IPv6 ranges that carry an IPv4 address in their last 32 bits: judge that address instead
    ipaddress.ip_network("::/96"),  # deprecated IPv4-compatible, e.g. ::127.0.0.1
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64: 64:ff9b::7f00:1 reaches 127.0.0.1 through a gateway
)


def _embedded_v4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    for network in _EMBEDDING:
        if ip in network:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip in ipaddress.ip_network("2002::/16"):  # 6to4: the IPv4 address sits in bits 16..48
        return ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
    return None


def is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv6Address) and (inner := _embedded_v4(ip)) is not None:
        return is_public_ip(str(inner))
    return ip.is_global and not ip.is_multicast


def require_public_url(url: str) -> None:
    """Raise ``BlockedAddress`` when the URL's host is not public; ``OSError`` when it does not resolve."""
    host = urlsplit(url).hostname or ""
    if not host:
        raise BlockedAddress(url)
    for info in socket.getaddrinfo(host, None):
        if not is_public_ip(info[4][0]):
            raise BlockedAddress(host)


def guard_needed(settings) -> bool:
    """Whether URLs from clients must be public: the server can be reached from other computers."""
    return not settings.is_loopback
