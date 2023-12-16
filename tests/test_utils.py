import sys
import pytest  # type: ignore
from pathlib import Path
from datetime import datetime, timedelta
from itertools import pairwise, accumulate
from functools import cached_property
from math import ceil
from typing import Generator, Any
from multiprocessing import Process
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from socketserver import ThreadingMixIn
from pydantic import BaseModel, Field
from enum import StrEnum, IntEnum
import click


from asyncio import Task, create_task, sleep, gather, timeout, TimeoutError
from random import random
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.resolve() / "src"))

from pydantic_exportables import Idx, JSONExportable, Importable, get_model
from pyutils.utils import ClickApp, epoch_now
from pyutils import ThrottledClientSession

logger = logging.getLogger()
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


HOST: str = "localhost"
PORT: int = 8889
MODEL_PATH: str = "/JSONParent"
RATE_FAST: float = 100
RATE_SLOW: float = 0.6

N_FAST: int = 500
N_SLOW: int = 5

THREADS: int = 5
# N : int = int(1e10)

logger = logging.getLogger()
message = logger.warning


class Eyes(StrEnum):
    blue = "Blue"
    grey = "Grey"
    brown = "Brown"


class Hair(IntEnum):
    black = 0
    brown = 1
    red = 2
    blonde = 3


class JSONChild(BaseModel):
    name: str
    created: int = Field(default_factory=epoch_now)

    @property
    def index(self) -> Idx:
        """return backend index"""
        return self.name

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        return {"name": self.index}


class JSONParent(JSONExportable, Importable):
    name: str
    amount: int = 0
    correct: bool = Field(default=False, alias="c")
    array: list[str] = list()
    child: JSONChild | None = None

    _exclude_unset = False

    @property
    def index(self) -> Idx:
        """return backend index"""
        return self.name

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        return {"name": self.index}


def json_data() -> list[JSONParent]:
    c1 = JSONChild(name="c1")
    c3 = JSONChild(name="c3")
    res: list[JSONParent] = list()
    res.append(JSONParent(name="P1", amount=1, array=["one", "two"], child=c1))
    res.append(JSONParent(name="P2", amount=-6, array=["three", "four"]))
    res.append(JSONParent(name="P3", amount=-6, child=c3))
    return res


class _HttpRequestHandler(BaseHTTPRequestHandler):
    """HTTPServer mock request handler"""

    _res_JSONParent: list[JSONParent] = json_data()

    @cached_property
    def url(self):
        return urlparse(self.path)

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Handle GET requests"""
        self.send_response(200)
        if self.url.path == MODEL_PATH:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            l: int = len(self._res_JSONParent)
            idx: int = epoch_now() % l
            res: JSONParent = self._res_JSONParent[idx]
            self.wfile.write(res.json_src().encode())

        else:
            self.send_header("Content-Type", "application/txt")
            self.end_headers()
            self.wfile.write(datetime.utcnow().isoformat().encode())

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Handle POST requests
        DOES NOT WORK YET"""
        message(f"POST @ {datetime.utcnow()}")
        self.send_response(200)
        self.send_header("Content-Type", "application/txt")
        self.end_headers()
        if self.url.path == MODEL_PATH:
            message(f"POST {self.url.path} @ {datetime.utcnow()}")
            if (
                _ := JSONParent.model_validate_json(self.rfile.read().decode())
            ) is not None:
                # assert False, "POST read content OK"
                message(f"POST OK @ {datetime.utcnow()}")
                self.wfile.write("OK".encode())
                # assert False, "POST did write"
            else:
                # assert False, "POST read content ERROR"
                message(f"POST ERROR @ {datetime.utcnow()}")
                self.wfile.write("ERROR".encode())
        # assert False, "do_POST()"

    def log_request(self, code=None, size=None):
        """Don't log anything"""
        pass


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


class _HttpServer(Process):
    def __init__(self, host: str = HOST, port: int = PORT):
        super().__init__()
        self._host: str = host
        self._port: int = port

    def run(self):
        server = ThreadingSimpleServer((self._host, self._port), _HttpRequestHandler)
        server.serve_forever()


def max_rate(timings: list[float], rate: float) -> float:
    """Read list[datetime] and return number of timings,
    average rate and maximum rate"""
    assert rate > 0, f"rate must be positive: {rate}"
    diffs: list[float] = [x1 - x0 for (x0, x1) in pairwise(timings)]
    cums: list[float] = [0] + list(accumulate(diffs))
    window: int = max(int(rate) - 1, 1)
    min_time: float = min(
        [cums[i + window] - cums[i] for i in range(len(cums) - window)]
    )
    return (window) / min_time


def avg_rate(timings: list[float]) -> float:
    n: int = len(timings) - 1  # the last request is not measured in total
    total: float = timings[-1] - timings[0]
    return n / total


async def _get(url: str, rate: float, N: int) -> list[float]:
    """Test timings of N/sec get"""
    timings: list[float] = list()
    async with ThrottledClientSession(rate_limit=rate) as session:
        for _ in range(N):
            async with session.get(url, ssl=False) as resp:
                assert resp.status == 200, f"request failed, HTTP STATUS={resp.status}"
                timings.append(datetime.fromisoformat(await resp.text()).timestamp())
    return timings


@pytest.fixture(scope="module")
def server_host() -> str:
    return HOST


@pytest.fixture(scope="module")
def server_port() -> int:
    return PORT


@pytest.fixture(scope="module")
def server_url(server_host: str, server_port: int) -> Generator[str, None, None]:
    # start HTTP server
    host: str = server_host
    port: int = server_port
    server: _HttpServer = _HttpServer(host=host, port=server_port)
    server.start()
    yield f"http://{host}:{port}/"
    # clean up
    server.terminate()


@pytest.fixture()
def model_path() -> str:
    return MODEL_PATH


@click.group(help="CLI tool test")
@click.option(
    "--normal",
    "LOG_LEVEL",
    flag_value=logging.WARNING,
    default=True,
    help="default verbosity",
)
@click.option("--verbose", "LOG_LEVEL", flag_value=logging.INFO, help="verbose logging")
@click.option("--debug", "LOG_LEVEL", flag_value=logging.DEBUG, help="debug logging")
@click.option(
    "--log", type=click.Path(path_type=Path), default=None, help="log to FILE"
)
@click.pass_context
def cli(
    ctx: click.Context,
    LOG_LEVEL: int = logging.WARNING,
    log: Path | None = None,
) -> None:
    """CLI app to extract WoT Blitz tankopedia and maps for other tools"""
    global logger, error, debug, verbose, message
    click.echo(f"LOG_LEVEL={LOG_LEVEL}, log={log}")


@cli.group(help="Test more")
@click.option(
    "-f",
    "--force",
    flag_value=True,
    default=False,
    help="Force testing",
)
@click.pass_context
def more(ctx: click.Context, force: bool = False) -> None:
    click.echo(f"more: force={force}")


@more.command(help="do test")
def do():
    click.echo("do")


@more.command(help="don't test")
def dont():
    click.echo("dont")


@cli.group(help="Test less")
@click.option(
    "-f",
    "--force",
    flag_value=True,
    default=False,
    help="Force testing",
)
@click.pass_context
def less(ctx: click.Context, force: bool = False) -> None:
    click.echo(f"less: force={force}")


@less.command(name="do", help="do test")
def do_less():
    click.echo("do")


@less.command(name="dont", help="don't test")
def dont_less():
    click.echo("dont")


def test_1_ClickApp() -> None:
    """Test ClickApp() helper"""
    app = ClickApp(cli, "test-app")
    app.add_command(["more"])
    app.add_command(["more", "do", "--help"])
    app.add_command(["more", "dont"])
    app.add_command(["less", "--help"])
    app.add_command(["less", "do", "--help"])
    app.add_command(["less", "dont"])

    docs: str = app.mk_docs()
    assert len(docs) > 100, "making docs failed"


@pytest.mark.timeout(20)
@pytest.mark.asyncio
async def test_2_get_model(server_url: str, model_path: str) -> None:
    """Test get_url_model()"""
    rate_limit: float = RATE_SLOW
    N: int = N_SLOW
    url: str = server_url + model_path
    res: JSONParent | None
    async with ThrottledClientSession(rate_limit=rate_limit) as session:
        for _ in range(N):
            if (
                _ := await get_model(
                    session=session, url=url, resp_model=JSONParent, retries=2
                )
            ) is None:
                assert False, "get_url_model() returned None"
