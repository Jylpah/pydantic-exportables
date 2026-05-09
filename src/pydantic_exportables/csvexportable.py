########################################################
#
# CSVExportable()
#
########################################################

import logging
from typing import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Type,
    Literal,
    Iterable,
    TypeVar,
)
from csv import Dialect, excel
from aiocsv import AsyncDictReader, AsyncDictWriter

from pathlib import Path
from typing import Any, Self
from pydantic import ConfigDict

import aiofiles

from datetime import date, datetime
from enum import Enum

from .jsonexportable import JSONExportable
from .utils import str2path

# Setup logging
logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


class CSVExportable(JSONExportable):
    """Abstract class to provide CSV export"""

    # # Define subclass' CSV readers/writers into these
    # _csv_custom_writers: ClassVar[MutableMapping[str, Callable[[Any], Any]]] = dict()
    # _csv_custom_readers: ClassVar[MutableMapping[str, Callable[[Any], Any]]] = dict()
    # # Do not store directly into these
    # _csv_writers: ClassVar[MutableMapping[str, Callable[[Any], Any]]] = dict()
    # _csv_readers: ClassVar[MutableMapping[str, Callable[[Any], Any]]] = dict()

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
    )

    # @classmethod
    # def __pydantic_init_subclass__(cls, **kwargs) -> None:
    #     """Use PEP 487 sub class constructor instead a custom one"""
    #     # makes sure each subclass has its own CSV field readers/writers.
    #     # Inherits the parents field functions using copy.deepcopy()
    #     super().__pydantic_init_subclass__(**kwargs)
    #     try:
    #         cls._csv_writers = cls._csv_writers.copy()  # type: ignore
    #         cls._csv_readers = cls._csv_readers.copy()  # type: ignore
    #     except AttributeError:
    #         cls._csv_writers = dict()
    #         cls._csv_readers = dict()
    #     cls._csv_writers.update(cls._csv_custom_writers)
    #     cls._csv_readers.update(cls._csv_custom_readers)

    def csv_headers(self, by_alias: bool = False, sep: str = ".") -> list[str]:
        """Provide CSV headers as list"""
        # should this be sorted to maintain column order?
        return list(self.flatten(sep=sep, by_alias=by_alias).keys())

    # def _csv_write_fields(
    #     self, left: dict[str, Any]
    # ) -> tuple[dict[str, Any], dict[str, Any]]:
    #     """Write CSV fields with custom encoders

    #     Returns columns_done, columns_left
    #     """
    #     res: dict[str, Any] = dict()
    #     # debug ("_csv_write_fields(): starting: %s", str(type(self)))
    #     # debug("Class: %s: csv_writers: %s", type(self), str(self._csv_writers))

    #     for field, encoder in self._csv_writers.items():
    #         # debug ("class=%s, field=%s, encoder=%s", str(type(self)), field, str(encoder))
    #         try:
    #             if left[field] is not None and left[field] != "":
    #                 res[field] = encoder(left[field])
    #             del left[field]
    #         except KeyError as err:
    #             debug("field=%s not found: %s", field, err)

    #     # debug("Class: %s: res: %s", type(self), str(res))
    #     # debug("Class: %s: left: %s", type(self), str(left))
    #     return res, left

    def csv_row(self) -> dict[str, str]:
        """
        CSVExportable._csv_row() takes care of str,int,float,bool,Enum, date and datetime.
        Class specific implementation needs to take care or serializing other fields.
        Custom fields are serialized using __str__() method, so make sure to implement it for custom fields in CSVExportable.
        """
        res: dict[str, Any] = dict()
        flattened: dict[str, Any] = self.flatten()
        for key, value in flattened.items():
            if isinstance(value, Enum):
                res[key] = value.value
            elif isinstance(value, date):
                res[key] = value.isoformat()
            elif isinstance(value, datetime):
                res[key] = value.isoformat()
            elif value is not None:
                res[key] = str(value)
            else:
                res[key] = ""
        return res

    # def _clear_None(
    #     self, res: dict[str, str | int | float | bool | None]
    # ) -> dict[str, str | int | float | bool]:
    #     out: dict[str, str | int | float | bool] = dict()
    #     for key, value in res.items():
    #         if value is None:
    #             out[key] = ""
    #         else:
    #             out[key] = value
    #     return out

    # @classmethod
    # def _csv_read_fields(
    #     cls, row: dict[str, Any]
    # ) -> tuple[dict[str, Any], dict[str, Any]]:
    #     """read CSV fields with custom encoding.
    #     Returns read, unread fields as dict[str, Any]"""

    #     res: dict[str, Any] = dict()
    #     # if cls is CSVExportable:
    #     #     return res, row
    #     # debug ("%s._csv_read_fields(): %s", cls.__name__, str(row))
    #     for field, decoder in cls._csv_readers.items():
    #         # debug (
    #         #     "%s._csv_read_fields(): field=%s, decoder=%s, value=%s", cls.__name__, field, str(decoder), row[field]
    #         # )
    #         try:
    #             if row[field] != "":
    #                 res[field] = decoder(row[field])
    #             del row[field]
    #         except KeyError:
    #             debug("field=%s not found", field)
    #     # debug ("class=%s", str(cls))

    #     return res, row

    @classmethod
    def from_csv(
        cls, row: dict[str, Any], by_alias: bool = False, sep: str = "."
    ) -> Self:
        """
        Create an instance of the class from a CSV row. Does not work with alias field names.

        Returns None if parsing fails.
        """
        try:
            if not isinstance(row, dict):
                raise TypeError("row has to be type 'dict'")
            keys: list[str] = list(row.keys())
            for key in keys:
                if row[key] == "":
                    row[key] = None
            return cls.from_flattened(row, sep=sep, by_alias=by_alias, from_str=True)
        except Exception as err:
            raise err

    @classmethod
    async def import_csv(
        cls,
        filename: Path | str,
        dialect: type[Dialect] = excel,
        by_alias: bool = False,
        sep: str = ".",
    ) -> AsyncGenerator[Self, None]:
        """
        Import from filename, one model per line.
        """
        debug("importing from CSV file: %s", str(filename))
        async with aiofiles.open(filename, mode="r", newline="") as f:
            async for row in AsyncDictReader(f, dialect=dialect):
                try:
                    yield cls.from_csv(row, by_alias=by_alias, sep=sep)
                except Exception as err:
                    error("Could read line: %s", err)

        # res: dict[str, Any]
        # # debug("from_csv(): trying to import from: %s", str(row))
        # res, row = cls._csv_read_fields(row)

        # for field in row.keys():
        #     if row[field] != "":
        #         try:
        #             if (field_type := cls.model_fields[field].annotation) is None:
        #                 field_type = str
        #             # debug ("field=%s, field_type=%s, value=%s", field, field_type, row[field])
        #             if field_type in {int, float, str}:
        #                 res[field] = (field_type)(str(row[field]))
        #             elif field_type is bool:
        #                 res[field] = row[field] == "True"
        #             elif issubclass(field_type, Enum):
        #                 res[field] = field_type[
        #                     str(row[field])
        #                 ]  ## Enums are stored by key, not value
        #             elif field_type is date:
        #                 res[field] = date.fromisoformat(row[field])
        #             elif field_type is datetime:
        #                 res[field] = datetime.fromisoformat(row[field])
        #             else:
        #                 res[field] = (field_type)(str(row[field]))
        #         except KeyError:  # field not in cls
        #             continue
        #         except AttributeError:
        #             error(f"Class {cls.__name__}() does not have attribute: {field}")
        #         except Exception:
        #             # debug ("%s raised, trying direct assignment: %s", type(err), err)
        #             res[field] = str(row[field])
        # try:
        #     # debug ("from_csv(): trying parse: %s", str(res))
        #     return cls.model_validate(res)
        # except ValidationError as err:
        #     error(f"Could not parse row ({row}): {err}")
        # return None


##########################################################
#
# CSV functions
#
##########################################################


# async def write_csv(
#     filename: Path | str,
#     items: Iterable[CSVExportable],
#     force: bool = False,
#     append: bool = False,
# ) -> tuple[int, int]:
#     exported: int = 0
#     errors: int = 0
#     try:
#         filename = str2path(filename, ".csv")
#         if filename.is_file() and (not (force or append)):
#             raise FileExistsError(f"Cannot export to {filename}")
#         mode: Literal["w", "a"] = "a" if append else "w"

#         debug("opening %s for writing in mode=%s", str(filename), mode)
#         async with aiofiles.open(filename, mode=mode, newline="") as csvfile:
#             try:
#                 header = items.
#                 writer = AsyncDictWriter(
#                     csvfile, fieldnames=fields, dialect=dialect
#                 )
#                 if not append:
#                     await writer.writeheader()
#             except Exception as err:
#                 error(err)
#                 raise

#             while exportable is not None:
#                 try:
#                     # debug(f'Writing row: {exportable.csv_row()}')
#                     await writer.writerow(exportable.csv_row())
#                     exported += 1
#                 except Exception as err:
#                     error(f"error writing CSV row type={type(exportable)}: {err}")
#                     errors += 1
#                 exportable = await anext(aiterator, None)

#     except OSError as err:
#         error(f"could not write to {filename}: {err}")
#         raise

#     return exported, errors


async def export_csv(
    filename: Path | str,
    iterable: AsyncIterable[CSVExportable] | Iterable[CSVExportable],
    force: bool = False,
    append: bool = False,
) -> tuple[int, int]:
    """
    Export data to a CSVfile
    If filename is "-", write to STDOUT."
    Returns a tuple of (rows exported, errors)
    """
    T = TypeVar("T")

    async def async_iter(iterable: Iterable[T]) -> AsyncIterable[T]:
        for item in iterable:
            yield item

    debug("starting")
    # assert isinstance(Q, Queue), "Q has to be type of asyncio.Queue[CSVExportable]"
    # assert type(filename) is str and len(filename) > 0, "filename has to be str"
    exported: int = 0
    errors: int = 0
    dialect: Type[Dialect] = excel
    if isinstance(iterable, Iterable):
        iterable = async_iter(iterable)
    aiterator: AsyncIterator[CSVExportable] = aiter(iterable)
    exportable: CSVExportable | None = await anext(aiterator, None)

    if exportable is None:
        raise ValueError("empty iterable given")
    fields: list[str] = exportable.csv_headers()

    if isinstance(filename, str) and filename == "-":  # STDOUT
        # print header
        print(dialect.delimiter.join(fields))
        while exportable is not None:
            try:
                row: dict[str, str] = exportable.csv_row()
                print(dialect.delimiter.join([row[key] for key in fields]))
                exported += 1
            except KeyError as err:
                error(f"row does not have field: {err}")
                errors += 1
            exportable = await anext(aiterator, None)
        debug("export finished")

    else:  # File
        try:
            filename = str2path(filename, ".csv")
            if filename.is_file() and (not (force or append)):
                raise FileExistsError(f"Cannot export to {filename}")
            mode: Literal["w", "a"] = "a" if append else "w"

            debug("opening %s for writing in mode=%s", str(filename), mode)
            async with aiofiles.open(filename, mode=mode, newline="") as csvfile:
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
                        exported += 1
                    except Exception as err:
                        error(f"error writing CSV row type={type(exportable)}: {err}")
                        errors += 1
                    exportable = await anext(aiterator, None)

        except OSError as err:
            error(f"could not write to {filename}: {err}")
            raise

    return exported, errors
