"""Лёгкий просмотрщик файлов с правильной кодировкой UTF-8."""
import http.server
import socketserver

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".md": "text/markdown; charset=utf-8",
        ".py": "text/plain; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".yaml": "text/plain; charset=utf-8",
    }

with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"Сервер файлов: http://127.0.0.1:{PORT}")
    httpd.serve_forever()