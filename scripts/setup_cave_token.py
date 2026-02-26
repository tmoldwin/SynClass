"""Save your CAVE token securely (no echo, never in command line or logs).

1. Get a token: https://global.daf-apis.com/auth/api/v1/create_token
   (Log in with Google, copy the token.)
2. Run: python scripts/setup_cave_token.py
3. Paste the token when prompted; it will not be shown on screen.

Token is written to your user secrets dir (~/.cloudvolume/secrets/), not in the repo.
"""
import sys

try:
    import getpass
except ImportError:
    getpass = None

from caveclient import CAVEclient

# Optional: read from a local file (gitignored). File is removed after use.
_LOCAL_TOKEN_FILE = ".cave_token"

def _read_token():
    import os
    if os.path.isfile(_LOCAL_TOKEN_FILE):
        with open(_LOCAL_TOKEN_FILE, "r") as f:
            tok = f.read().strip()
        try:
            os.remove(_LOCAL_TOKEN_FILE)
        except OSError:
            pass
        return tok
    if getpass:
        return getpass.getpass("Paste CAVE token (no echo): ").strip()
    return input("Paste CAVE token: ").strip()

def main():
    token = _read_token()
    if not token:
        print("No token provided. Exiting.")
        sys.exit(1)
    # Use same server as minnie65_public so token is stored for that endpoint
    client = CAVEclient(server_address="https://global.daf-apis.com")
    client.auth.save_token(token=token, overwrite=True)
    del token
    print("Token saved to user secrets (not in repo).")
    try:
        CAVEclient("minnie65_public")
        print("OK: minnie65_public connection works.")
    except Exception as e:
        print("Check failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
