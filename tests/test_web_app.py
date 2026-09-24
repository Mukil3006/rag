"""Local HTTP boundary tests; inference is exercised separately through the browser."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from app import ResearchApp, make_handler


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=ResearchApp()
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.app))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.app.worker.shutdown()

    def test_static_ui_is_served_with_no_external_scripts(self):
        with urlopen(self.url) as response:
            text=response.read().decode()
            self.assertIn('Complete terminal output',text)
            self.assertIn('frame-ancestors',response.headers['Content-Security-Policy'])

    def test_invalid_question_never_starts_inference(self):
        request=Request(self.url+'/api/runs',data=json.dumps({'query':'','mode':'C'}).encode(),headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as ctx:urlopen(request)
        self.assertEqual(ctx.exception.code,400)
        self.assertIsNone(self.app.active)

    def test_cross_origin_run_is_blocked(self):
        request=Request(self.url+'/api/runs',data=b'{}',headers={'Origin':'https://example.com','Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as ctx:urlopen(request)
        self.assertEqual(ctx.exception.code,403)

    def test_arbitrary_project_files_are_not_served(self):
        for path in ('/src/config.py','/api/saved/../../src/config.py'):
            with self.assertRaises(HTTPError) as ctx:urlopen(self.url+path)
            self.assertEqual(ctx.exception.code,404)


if __name__=='__main__':unittest.main()
