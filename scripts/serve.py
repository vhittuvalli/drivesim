# Dev server: like `python3 -m http.server`, but tells the browser not to cache, so edited
# modules are always reloaded (otherwise a stale main.js can run against a new index.html).
import http.server
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
http.server.ThreadingHTTPServer(('', port), NoCache).serve_forever()
