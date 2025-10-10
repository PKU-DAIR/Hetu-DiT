import socket


def get_loopback_host():
    try:
        socket.inet_pton(socket.AF_INET6, "::1")
        return "::1"  # IPv6 loopback is available
    except OSError:
        return "127.0.0.1"  # fallback to IPv4
