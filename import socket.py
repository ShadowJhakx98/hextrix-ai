import socket
try:
    print(f"Resolving Cloudflare API: {socket.gethostbyname('api.cloudflare.com')}")
    print(f"Resolving Google API: {socket.gethostbyname('generativelanguage.googleapis.com')}")
except Exception as e:
    print(f"DNS resolution test failed: {e}")