import logging
from typing import (
    Type,
    Literal,
    AsyncIterable,
    AsyncIterator,
)
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from asyncio import CancelledError
from aiofiles import open
from os import linesep
from aiocsv.writers import AsyncDictWriter
from csv import Dialect, excel
from abc import abstractmethod

from eventcounter import EventCounter
from .jsonexportable import JSONExportable
from .csvexportable import CSVExportable
from .utils import str2path

# Setup logging
logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug

########################################################
#
# TXTExportable()
#
########################################################


class TXTExportable(BaseModel):
    """Abstract class to provide TXT export"""

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
    )

    @abstractmethod
    def txt_row(self, format: str = "") -> str:
        """export data as single row of text"""
        raise NotImplementedError


########################################################
#
# CSVExportable()
#
########################################################


EXPORT_FORMAT = Literal["txt", "json", "csv"]
EXPORT_FORMATS = ["txt", "json", "csv"]


async def export_csv(
    iterable: AsyncIterable[CSVExportable],
    filename: Path | str,
    force: bool = False,
    append: bool = False,
) -> EventCounter:
    """Export data to a CSVfile"""
    debug("starting")
    # assert isinstance(Q, Queue), "Q has to be type of asyncio.Queue[CSVExportable]"
    # assert type(filename) is str and len(filename) > 0, "filename has to be str"
    stats: EventCounter = EventCounter("export CSV")

    dialect: Type[Dialect] = excel
    aiterator: AsyncIterator[CSVExportable] = aiter(iterable)
    exportable: CSVExportable | None = await anext(aiterator, None)

    if exportable is None:
        debug("empty iterable given")
        return stats
    fields: list[str] = exportable.csv_headers()

    if isinstance(filename, str) and filename == "-":  # STDOUT
        try:
            # print header
            print(dialect.delimiter.join(fields))
            while exportable is not None:
                try:
                    row: dict[str, str | int | float | bool] = exportable.csv_row()
                    print(dialect.delimiter.join([str(row[key]) for key in fields]))
                    stats.log("rows")
                except KeyError as err:
                    error(f"CSVExportable object does not have field: {err}")
                    stats.log("errors")
                except CancelledError:
                    raise
                exportable = await anext(aiterator, None)

            debug("export finished")
        except CancelledError:
            debug("Cancelled")
            raise

    else:  # File
        try:
            filename = str2path(filename, ".csv")
            if filename.is_file() and (not (force or append)):
                raise FileExistsError(f"Cannot export to {filename}")
            mode: Literal["w", "a"] = "a" if append else "w"

            debug("opening %s for writing in mode=%s", str(filename), mode)
            async with open(filename, mode=mode, newline="") as csvfile:
                try:
                    writer = AsyncDictWriter(
                        csvfile, fieldnames=fields, dialect=dialect
                    )
                    if not append:
                        await writer.writeheader()
                except Exception as err:
                    error(err)
                    raise

                while exportable is not None:
                    try:
                        # debug(f'Writing row: {exportable.csv_row()}')
                        await writer.writerow(exportable.csv_row())
                        stats.log("rows")
                    except CancelledError:
                        raise
                    except Exception as err:
                        error(f"error writing CSV row type={type(exportable)}: {err}")
                        stats.log("errors")
                    exportable = await anext(aiterator, None)

        except CancelledError:
            debug("Cancelled")
            raise
        except OSError as err:
            error(f"could not write to {filename}: {err}")
            raise
        except Exception as err:
            error(f"error exporting to CSV: {err}")
            raise
    return stats


async def export_json(
    iterable: AsyncIterable[JSONExportable],
    filename: Path | str,
    force: bool = False,
    append: bool = False,
) -> EventCounter:
    """Export data to a JSON file"""
    # assert type(filename) is str and len(filename) > 0, "filename has to be str"
    stats: EventCounter = EventCounter("export JSON")
    try:
        exportable: JSONExportable
        if isinstance(filename, str) and filename == "-":  # STDOUT
            async for exportable in iterable:
                try:
                    print(exportable.json_src(indent=4))
                    stats.log("rows")
                except CancelledError:
                    raise
                except Exception as err:
                    error(f"error exporting JSON type={type(exportable)}: {err}")
                    stats.log("errors")
        else:  # FILE
            filename = str2path(filename, ".json")
            if filename.is_file() and (not (force or append)):
                raise FileExistsError(f"Cannot export to {filename}")
            mode: Literal["w", "a"] = "a" if append else "w"

            debug("opening %s for writing in mode=%s", str(filename), mode)
            async with open(filename, mode=mode) as txtfile:
                async for exportable in iterable:
                    try:
                        debug("writing JSON: %s", exportable.json_src())
                        await txtfile.write(exportable.json_src() + linesep)
                        stats.log("rows")
                    except CancelledError:
                        raise
                    except Exception as err:
                        error(f"{err}")
                        stats.log("errors")

    except CancelledError:
        debug("Cancelled")
        raise
    except Exception as err:
        error(f"error exporting to JSON: {err}")
        raise
    return stats


async def export_txt(
    iterable: AsyncIterable[TXTExportable],
    filename: Path | str,
    force: bool = False,
    append: bool = False,
) -> EventCounter:
    """Export data to a text file"""
    # assert type(filename) is str and len(filename) > 0, "filename has to be str"
    stats: EventCounter = EventCounter("export text")
    try:
        exportable: TXTExportable
        if isinstance(filename, str) and filename == "-":
            async for exportable in iterable:
                try:
                    print(exportable.txt_row(format="rich"))
                    stats.log("rows")
                except Exception as err:
                    error(f"{err}")
                    stats.log("errors")
        else:
            filename = str2path(filename, ".txt")
            if filename.is_file() and (not (force or append)):
                raise FileExistsError(f"Cannot export to {filename}")
            mode: Literal["w", "a"] = "a" if append else "w"

            async with open(filename, mode=mode) as txtfile:
                async for exportable in iterable:
                    try:
                        await txtfile.write(exportable.txt_row() + linesep)
                        stats.log("rows")
                    except Exception as err:
                        error(f"error exporting to text type={type(exportable)}: {err}")
                        stats.log("errors")

    except CancelledError:
        debug("Cancelled")
    except Exception as err:
        error(f"error exporting to text: {err}")
        raise
    return stats


async def export(
    iterable: AsyncIterable[CSVExportable]
    | AsyncIterable[TXTExportable]
    | AsyncIterable[JSONExportable],
    format: EXPORT_FORMAT,
    filename: Path | str,
    force: bool = False,
    append: bool = False,
) -> EventCounter:
    """Export data to file or STDOUT"""
    debug("starting")
    stats: EventCounter = EventCounter("export")

    # if filename != "-":
    #     for export_format in EXPORT_FORMATS:
    #         if filename.endswith(export_format) and export_format in get_args(EXPORT_FORMAT):
    #             format = cast(EXPORT_FORMAT, export_format)

    try:
        if format == "txt":
            stats.merge(
                await export_txt(
                    iterable,  # type: ignore
                    filename=filename,
                    force=force,
                    append=append,
                )
            )
        elif format == "json":
            stats.merge(
                await export_json(
                    iterable,  # type: ignore
                    filename=filename,
                    force=force,
                    append=append,
                )
            )
        elif format == "csv":
            stats.merge(
                await export_csv(
                    iterable,  # type: ignore
                    filename=filename,
                    force=force,
                    append=append,
                )
            )  # type: ignore
        else:
            raise ValueError(f"Unknown format: {format}")
    except Exception as err:
        stats.log("errors")
        error(f"{err}")
        raise
    return stats
