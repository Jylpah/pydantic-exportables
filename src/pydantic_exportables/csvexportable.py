########################################################
#
# CSVExportable()
#
########################################################

import logging
from typing import Any, Self
from pydantic import ConfigDict

from datetime import date, datetime
from enum import Enum

from result import Result, Ok, Err

from .jsonexportable import JSONExportable

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

    def csv_headers(
        self, by_alias: bool = False, sep: str = "."
    ) -> Result[list[str], str]:
        """Provide CSV headers as list"""
        try:
            return Ok(
                # list(self.model_dump(exclude_unset=False, by_alias=by_alias).keys())
                list(
                    self.flatten(sep=sep, by_alias=by_alias).keys()
                )  # should this be sorted to maintain column order?
            )
        except Exception as e:
            return Err(str(e))

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

    def csv_row(self) -> Result[dict[str, str], str]:
        """
        CSVExportable._csv_row() takes care of str,int,float,bool,Enum, date and datetime.
        Class specific implementation needs to take care or serializing other fields.
        Custom fields are serialized using __str__() method, so make sure to implement it for custom fields in CSVExportable.
        """
        res: dict[str, Any] = dict()
        try:
            flattened: dict[str, Any] = self.flatten()
            for key, value in flattened.items():
                if isinstance(value, Enum):
                    res[key] = value.name
                elif isinstance(value, date):
                    res[key] = value.isoformat()
                elif isinstance(value, datetime):
                    res[key] = value.isoformat()
                elif value is not None:
                    res[key] = str(value)
                else:
                    res[key] = ""
            return Ok(res)
        except Exception as err:
            return Err(str(err))

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
        cls, row: dict[str, Any], by_alias: bool = False, sep: str = ","
    ) -> Result[Self, str]:
        """
        Create an instance of the class from a CSV row. Does not work with alias field names.

        Returns None if parsing fails.
        """
        try:
            if not isinstance(row, dict):
                return Err("row has to be type dict()")
            return Ok(cls.from_flattened(row, sep=sep, by_alias=by_alias))
        except Exception as err:
            return Err(str(err))

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
