"""Observe synthetic Pi traffic at the loopback boundary; never forward auth headers."""
import http.client, http.server, json, pathlib, time

def make_proxy(out):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            if self.path!='/v1/chat/completions': self.send_error(404); return
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            n=len(list(out.glob('request-*.json')))
            (out/f'request-{n}.json').write_text(json.dumps(payload))
            conn=http.client.HTTPConnection('127.0.0.1',18089,timeout=600)
            t=time.perf_counter(); first=None; timing={}; usage={}; finish=None; error=None
            try:
                conn.request('POST',self.path,json.dumps(payload),{'Content-Type':'application/json'})
                resp=conn.getresponse()
                self.send_response(resp.status); self.send_header('Content-Type',resp.getheader('Content-Type','text/event-stream')); self.end_headers()
                for line in resp:
                    self.wfile.write(line); self.wfile.flush()
                    if not line.startswith(b'data:') or line[5:].strip()==b'[DONE]': continue
                    event=json.loads(line[5:])
                    choice=(event.get('choices') or [{}])[0]
                    if any(choice.get('delta',{}).get(k) for k in ('reasoning_content','reasoning','content','tool_calls')) and first is None: first=time.perf_counter()
                    timing=event.get('timings') or timing
                    usage=event.get('usage') or usage
                    finish=choice.get('finish_reason') or finish
            except (BrokenPipeError,ConnectionResetError) as exc: error=type(exc).__name__
            finally:
                end=time.perf_counter(); conn.close()
                result=dict(timings=timing,usage=usage,ttft=first-t if first else None,wall=end-t,finish=finish,error=error)
                if first and usage.get('completion_tokens',0)>1:
                    result['stream_tps']=(usage['completion_tokens']-1)/(end-first)
                (out/f'response-{n}.json').write_text(json.dumps(result))
    return http.server.ThreadingHTTPServer(('127.0.0.1',18090),Handler)
