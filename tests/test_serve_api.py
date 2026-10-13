#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

ROOT = '/home/hatch/workspace/Litetorch-'
sys.path.insert(0, os.path.join(ROOT, 'python'))
SERVE = os.path.join(ROOT, 'tools', 'lt-serve')
PORT = 18080
BASE = f'http://127.0.0.1:{PORT}'

passed = failed = 0
def check(cond, name):
    global passed, failed
    if cond:
        passed += 1; print("PASS", name)
    else:
        failed += 1; print("FAIL", name)

def http_get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, r.read().decode()

def http_post(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def http_post_stream(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    chunks = []
    with urllib.request.urlopen(req, timeout=60) as r:
        while True:
            line = r.readline()
            if not line:
                break
            chunks.append(line.decode())
    return chunks

def main():
    if not os.path.exists(SERVE):
        print("lt-serve binary missing, building...")
        r = subprocess.run(['bash', os.path.join(ROOT, 'tools', 'build_lt_serve.sh')],
                           capture_output=True, text=True, timeout=600)
        print(r.stdout[-500:] if r.stdout else "", r.stderr[-500:] if r.stderr else "")
        assert r.returncode == 0 and os.path.exists(SERVE), "lt-serve build failed"

    tmpd = tempfile.mkdtemp(prefix='ltserve-')
    vocab = os.path.join(tmpd, 'vocab.txt')
    with open(vocab, 'w') as f:
        for i in range(64):
            f.write(f"tok{i}\n")

    proc = subprocess.Popen(
        [SERVE, '--port', str(PORT), '--arch', 'gpt2b-tiny',
         '--vocab', vocab, '--max-new-tokens', '8', '--threads', '2'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(50):
            try:
                s, _ = http_get('/health')
                if s == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not start")

        s, b = http_get('/health')
        check(s == 200 and json.loads(b)['status'] == 'ok', "GET /health")

        s, b = http_get('/v1/models')
        d = json.loads(b)
        check(s == 200 and d['object'] == 'list' and len(d['data']) == 1, "GET /v1/models")

        s, b = http_post('/v1/completions', {'prompt': 'tok1 tok2 tok3', 'max_tokens': 5})
        d = json.loads(b)
        check(s == 200 and d['object'] == 'text_completion', "completions object")
        check(isinstance(d['choices'][0]['text'], str), "completions text is str")
        check(d['choices'][0]['finish_reason'] in ('stop', 'length'), "completions finish_reason")
        u = d['usage']
        check(u['prompt_tokens'] == 3 and u['completion_tokens'] == 5
              and u['total_tokens'] == 8, "completions usage counts")

        s, b = http_post('/v1/completions', {'prompt': 'tok1 tok2', 'max_tokens': 4,
                                             'temperature': 0.7, 'top_p': 0.9, 'seed': 42})
        d = json.loads(b)
        n = d['usage']['completion_tokens']
        check(s == 200 and 1 <= n <= 4, "completions sampling params")

        chunks = http_post_stream('/v1/completions', {'prompt': 'tok5', 'max_tokens': 4,
                                                      'stream': True})
        data_lines = [c for c in chunks if c.startswith('data:')]
        check(len(data_lines) >= 2, "completions SSE has data lines")
        check(data_lines[-1].strip() == 'data: [DONE]', "completions SSE ends with [DONE]")
        mid = json.loads(data_lines[0][5:])
        check('choices' in mid, "completions SSE chunk schema")

        s, b = http_post('/v1/chat/completions',
                         {'messages': [{'role': 'user', 'content': 'tok1 tok2'}],
                          'max_tokens': 4})
        d = json.loads(b)
        check(s == 200 and d['object'] == 'chat.completion', "chat object")
        check(d['choices'][0]['message']['role'] == 'assistant', "chat role")
        check(isinstance(d['choices'][0]['message']['content'], str), "chat content")

        chunks = http_post_stream('/v1/chat/completions',
                                  {'messages': [{'role': 'user', 'content': 'tok3'}],
                                   'max_tokens': 3, 'stream': True})
        data_lines = [c for c in chunks if c.startswith('data:')]
        check(data_lines[-1].strip() == 'data: [DONE]', "chat SSE ends with [DONE]")
        first = json.loads(data_lines[0][5:])
        check(first['choices'][0]['delta'].get('role') == 'assistant', "chat SSE role delta")

        s, b = http_post('/v1/completions', {'max_tokens': 4})
        check(s == 400, "completions missing prompt -> 400")

        s, b = http_post('/v1/completions', 'not json {{{')
        check(s == 400, "completions invalid json -> 400")

        s, b = http_post('/v1/chat/completions', {'messages': 'nope'})
        check(s == 400, "chat bad messages -> 400")

        try:
            http_get('/nope')
            check(False, "unknown path -> 404")
        except urllib.error.HTTPError as e:
            check(e.code == 404, "unknown path -> 404")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"TOTAL: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
