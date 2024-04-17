import logging
from typing import (
    TypeVar,
    Self,
    AsyncGenerator,
)
from abc import ABCMeta
from pydantic import BaseModel, ValidationError
from aiofiles import open
from pathlib import Path

from .jsonexportable import JSONExportable
from .csvexportable import CSVExportable
from pyutils.utils import str2path

# Setup logging
logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


########################################################
#
# Importable()
#
########################################################


class Importable(metaclass=ABCMeta):
    """Abstract class to provide import"""

    @classmethod
    async def import_file(
        cls,
        filename: Path | str,
        **kwargs,
    ) -> AsyncGenerator[Self, None]:
        """Import models from a file, one per line"""
        debug("starting")
        filename = str2path(filename=filename)
        # try:
        if filename.name.lower().endswith(".txt") and issubclass(cls, TXTImportable):
            debug("importing from TXT file: %s", str(filename))
            async for obj in cls.import_txt(filename, **kwargs):
                yield obj
        elif filename.name.lower().endswith(".json") and issubclass(
            cls, JSONExportable
        ):
            debug("importing from JSON file: %s", str(filename))
            async for obj in cls.import_json(filename, **kwargs):
                yield obj
        elif filename.name.lower().endswith(".csv") and issubclass(cls, CSVExportable):
            debug("importing from CSV file: %s", str(filename))
            async for obj in cls.import_csv(filename):
                yield obj
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            yield
        # except Exception as err:
        #     error(f"{err}")

    @classmethod
    async def count_file(cls, filename: Path | str, **kwargs) -> int:
        """Count Importables in the file"""
        res: int = 0
        async for _ in cls.import_file(filename=filename, **kwargs):
            res += 1
        return res


########################################################
#
# TXTImportable()
#
########################################################


TXTImportableSelf = TypeVar("TXTImportableSelf", bound="TXTImportable")


class TXTImportable(BaseModel):
    """Abstract class to provide TXT import"""

    @classmethod
    def from_txt(cls, text: str, **kwargs) -> Self:
        """Provide parse object from a line of text"""
        raise NotImplementedError

    @classmethod
    async def import_txt(
        cls, filename: Path | str, **kwargs
    ) -> AsyncGenerator[Self, None]:
        """Import from filename, one model per line"""
        # try:
        debug(f"starting: {filename}")
        async with open(filename, "r") as f:
            async for line in f:
                try:
                    debug("line: %s", line)
                    if (
                        importable := cls.from_txt(line.rstrip(), **kwargs)
                    ) is not None:
                        yield importable
                except ValidationError as err:
                    error(f"Could not validate mode: {err}")
                except Exception as err:
                    error(f"{err}")
        # except Exception as err:
        #     error(f"Error importing file {filename}: {err}")
