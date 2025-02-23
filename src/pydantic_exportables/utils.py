import logging

from typing import Optional, TypeVar
from aiohttp import ClientSession, ClientError
from pydantic import BaseModel
from pathlib import Path
from asyncio import CancelledError, sleep


# Setup logging
logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug

# Constants
MAX_RETRIES: int = 3
SLEEP: float = 1

T = TypeVar("T")


##############################################
#
## Functions
#
##############################################


M = TypeVar("M", bound=BaseModel)


def str2path(filename: str | Path, suffix: str | None = None) -> Path:
    """convert filename (str) to pathlib.Path"""
    if isinstance(filename, str):
        filename = Path(filename)
    if suffix is not None and not filename.name.lower().endswith(suffix):
        filename = filename.with_suffix(suffix)
    return filename


async def get_model(
    session: ClientSession, url: str, resp_model: type[M], retries: int = MAX_RETRIES
) -> Optional[M]:
    """Get JSON from URL and return object. Validate JSON against resp_model, if given."""
    assert session is not None, "session cannot be None"
    assert url is not None, "url cannot be None"
    content: str | None = None
    try:
        if (
            content := await get_url(session=session, url=url, retries=retries)
        ) is not None:
            return resp_model.model_validate_json(content)
        debug("get_url() returned None")
    except ValueError as err:
        debug(
            f"{resp_model.__name__}: {url}: response={content}: Validation error={err}"
        )
    except Exception as err:
        debug(f"Unexpected error: {err}")
    return None


async def get_url(
    session: ClientSession, url: str, retries: int = MAX_RETRIES
) -> str | None:
    """Retrieve (GET) an URL and return content as text"""
    assert session is not None, "Session must be initialized first"
    assert url is not None, "url cannot be None"

    # if not is_url(url):
    #     raise ValueError(f"URL is malformed: {url}")

    for retry in range(1, retries + 1):
        debug(f"GET {url} try {retry} / {retries}")
        try:
            async with session.get(url) as resp:
                debug(f"GET {url} HTTP response status {resp.status}/{resp.reason}")
                if resp.ok:
                    return await resp.text()
        except ClientError as err:
            debug(f"Could not retrieve URL: {url} : {err}")
        except CancelledError as err:
            debug(f"Cancelled while still working: {err}")
            raise
        # except Exception as err:
        #     debug(f"Unexpected error {err}")
        await sleep(SLEEP)
    verbose(f"Could not retrieve URL: {url}")
    return None
