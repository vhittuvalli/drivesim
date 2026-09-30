# Dev server: static files with caching disabled (so edited modules always reload), plus a
# small API the browser uses to stream training data to disk:
#   POST /api/collect?run=<id>&name=<path>[&append=1]   write (or append to) data/<id>/<path>
#   GET  /api/datasets                                  runs in data/ with their frame counts
import http.server
import json
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
SAFE = re.compile(r'^[\w.-]+(/[\w.-]+)*$')


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        if not self.path.startswith('/api/collect'):  # one line per frame would drown the log
            super().log_message(fmt, *args)

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == '/api/datasets':
            runs = []
            if os.path.isdir(DATA):
                for run in sorted(os.listdir(DATA)):
                    samples = os.path.join(DATA, run, 'samples.jsonl')
                    if os.path.isfile(samples):
                        with open(samples) as f:
                            runs.append({'run': run, 'frames': sum(1 for _ in f)})
            return self.reply(200, {'runs': runs, 'frames': sum(r['frames'] for r in runs)})
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/api/collect':
            return self.reply(404, {'error': 'not found'})
        q = parse_qs(url.query)
        run, name = q.get('run', [''])[0], q.get('name', [''])[0]
        if not SAFE.match(run) or '/' in run or not SAFE.match(name) or '..' in name:
            return self.reply(400, {'error': 'bad run or name'})
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        path = os.path.join(DATA, run, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'ab' if q.get('append') == ['1'] else 'wb') as f:
            f.write(body)
        return self.reply(200, {'ok': True})


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f'Serving {ROOT} on http://localhost:{port}')

class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 256  # several browsers streaming frames at once
    daemon_threads = True


Server(('', port), Handler).serve_forever()
