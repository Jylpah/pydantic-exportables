import logging

from typing import Optional, TypeVar
from aiohttp import ClientSession
from pydantic import BaseModel

from pyutils.utils import get_url


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
