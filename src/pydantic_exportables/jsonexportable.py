########################################################
#
# JSONExportable()
#
########################################################

import logging
from typing import (
    Optional,
    Type,
    Any,
    Self,
    Literal,
    TypeVar,
    ClassVar,
    Union,
    Callable,
    Sequence,
    AsyncGenerator,
)
from pathlib import Path
from collections.abc import MutableMapping
from pydantic import BaseModel, ValidationError, ConfigDict
from aiofiles import open
from bson.objectid import ObjectId

from pyutils.utils import str2path

# Setup logging
logger = logging.getLogger()
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug

TypeExcludeDict = MutableMapping[int | str, Any]

# D = TypeVar("D", bound="JSONExportable")
# J = TypeVar("J", bound="JSONExportable")
# O = TypeVar("O", bound="JSONExportable")

DESCENDING: Literal[-1] = -1
ASCENDING: Literal[1] = 1
TEXT: Literal["text"] = "text"

Idx = Union[str, int, ObjectId]
BackendIndexType = Literal[-1, 1, "text"]
BackendIndex = tuple[str, BackendIndexType]
IdxType = TypeVar("IdxType", bound=Idx)


class JSONExportable(BaseModel):
    """Base class for Pydantic models with fail-safe JSON import & export and
    registrable model transformations. Returns None if parsing / importing / transformation fails
    """

    # fmt: off
    _exclude_export_DB_fields:  ClassVar[Optional[TypeExcludeDict]] = None
    _exclude_export_src_fields: ClassVar[Optional[TypeExcludeDict]] = None
    _include_export_DB_fields:  ClassVar[Optional[TypeExcludeDict]] = None
    _include_export_src_fields: ClassVar[Optional[TypeExcludeDict]] = None
    _export_DB_by_alias:    ClassVar[bool] = True
    _exclude_defaults:      ClassVar[bool] = True
    _exclude_unset:         ClassVar[bool] = True
    _exclude_none:          ClassVar[bool] = True
    _example:               ClassVar[str]  = ""
    # fmt: on

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
    )

    # This is set in every subclass using __init_subclass__()
    _transformations: ClassVar[
        MutableMapping[Type, Callable[[Any], Optional[Self]]]
    ] = dict()

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Workaround to be able to set fields without validation."""
        attr = getattr(self.__class__, name, None)
        if isinstance(attr, property):
            attr.__set__(self, value)
        else:
            self.__dict__[name] = value
            self.__pydantic_fields_set__.add(name)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        """Use PEP 487 sub class constructor instead a custom one"""
        # make sure each subclass has its own transformation register
        super().__pydantic_init_subclass__(**kwargs)
        cls._transformations = dict()

    @classmethod
    def register_transformation(
        cls,
        obj_type: Any,
        method: Callable[[Any], Optional[Self]],
    ) -> None:
        """Register transformations"""
        cls._transformations[obj_type] = method
        return None

    @classmethod
    def transform(cls, in_obj: Any) -> Optional[Self]:
        """Transform object to out_type if supported"""
        try:
            if type(in_obj) is cls:
                return in_obj
            else:
                return cls._transformations[type(in_obj)](in_obj)  # type: ignore
        except Exception as err:
            error(f"failed to transform {type(in_obj)} to {cls}: {err}")
        return None

    @classmethod
    def transform_many(cls, in_objs: Sequence[Any]) -> list[Self]:
        """Transform a Sequence of objects into list of Self"""
        return [out for obj in in_objs if (out := cls.transform(obj)) is not None]

    @classmethod
    def from_obj(
        cls, obj: Any, in_type: type[BaseModel] | None = None
    ) -> Optional[Self]:
        """Parse instance from raw object.
        Returns None if reading from object failed.
        """
        obj_in: BaseModel
        if in_type is None:
            try:
                return cls.model_validate(obj)
            except ValidationError as err:
                error("could not parse object as %s: %s", cls.__name__, str(err))
        else:
            try:
                if (obj_in := in_type.model_validate(obj)) is not None:
                    return cls.transform(obj_in)
            except ValidationError as err:
                error("could not parse object as %s: %s", cls.__name__, str(err))
        return None

    @classmethod
    def from_objs(
        cls, objs: Sequence[Any], in_type: type[BaseModel] | None = None
    ) -> list[Self]:
        """Parse list of instances from raw objects.
        Parsing failures are ignored silently.
        """
        return [
            out
            for obj in objs
            if (out := cls.from_obj(obj, in_type=in_type)) is not None
        ]

    @classmethod
    async def open_json(
        cls, filename: Path | str, exceptions: bool = False
    ) -> Self | None:
        """Open replay JSON file and return class instance"""
        try:
            async with open(filename, "r") as f:
                return cls.model_validate_json(await f.read())
        except ValueError as err:
            debug(f"Could not parse {type(cls)} from file: {filename}: {err}")
            if exceptions:
                raise
        except OSError as err:
            debug(f"Error reading file: {filename}: {err}")
            if exceptions:
                raise
        return None

    @classmethod
    def parse_str(cls, content: str) -> Self | None:
        """return class instance from a JSON string"""
        try:
            return cls.model_validate_json(content)
        except ValueError as err:
            debug(f"Could not parse {type(cls)} from JSON: {err}")
        return None

    @classmethod
    async def import_json(
        cls, filename: Path | str, **kwargs
    ) -> AsyncGenerator[Self, None]:
        """Import models from filename, one model per line"""
        try:
            # importable : JSONImportableSelf | None
            async with open(filename, "r") as f:
                async for line in f:
                    try:
                        if (importable := cls.parse_str(line, **kwargs)) is not None:
                            yield importable
                    except Exception as err:
                        error(f"{err}")
        except OSError as err:
            error(f"Error importing file {filename}: {err}")

    def _export_helper(
        self, params: dict[str, Any], fields: list[str] | None = None, **kwargs
    ) -> dict:
        """Helper func to process params for obj/src export funcs"""
        if fields is not None:
            del params["exclude"]
            params["include"] = {f: True for f in fields}
            params["exclude_defaults"] = False
            params["exclude_unset"] = False
            params["exclude_none"] = False
        elif "exclude" in kwargs:
            try:
                params["exclude"].update(kwargs["exclude"])
                del kwargs["exclude"]
            except Exception:
                pass
        elif "include" in kwargs:
            try:
                params["include"].update(kwargs["include"])
                del kwargs["include"]
            except Exception:
                pass
        # else:
        #     for f in ["exclude", "include"]:
        #         try:
        #             params[f].update(kwargs[f])
        #             del kwargs[f]
        #         except:
        #             pass
        params.update(kwargs)
        return params

    @property
    def index(self) -> Idx:
        """return backend index"""
        raise NotImplementedError

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        raise NotImplementedError

    @classmethod
    def backend_indexes(cls) -> list[list[tuple[str, BackendIndexType]]]:
        """return backend search indexes"""
        raise NotImplementedError

    @classmethod
    def example_instance(cls) -> Self:
        """return a example instance of the class"""
        if len(cls._example) > 0:
            return cls.model_validate_json(cls._example)
        raise NotImplementedError

    def __hash__(self) -> int:
        """Make object hashable using index fields if defined"""
        try:
            return hash(self.index)
        except NotImplementedError:
            return hash(id(self))

    def obj_db(self, fields: list[str] | None = None, **kwargs) -> dict:
        params: dict[str, Any] = {
            "exclude": self._exclude_export_DB_fields,
            "include": self._include_export_DB_fields,
            "exclude_defaults": self._exclude_defaults,
            "by_alias": self._export_DB_by_alias,
        }
        params = self._export_helper(params=params, fields=fields, **kwargs)
        return self.model_dump(**params)

    def obj_src(self, fields: list[str] | None = None, **kwargs) -> dict:
        params: dict[str, Any] = {
            "exclude": self._exclude_export_src_fields,
            "include": self._include_export_src_fields,
            "exclude_unset": self._exclude_unset,
            "exclude_none": self._exclude_none,
            "by_alias": not self._export_DB_by_alias,
        }
        params = self._export_helper(params=params, fields=fields, **kwargs)
        return self.model_dump(**params)

    def json_db(self, fields: list[str] | None = None, **kwargs) -> str:
        params: dict[str, Any] = {
            "exclude": self._exclude_export_DB_fields,
            "include": self._include_export_DB_fields,
            "exclude_defaults": self._exclude_defaults,
            "by_alias": self._export_DB_by_alias,
        }
        params = self._export_helper(params=params, fields=fields, **kwargs)
        return self.model_dump_json(**params)

    def json_src(self, fields: list[str] | None = None, **kwargs) -> str:
        params: dict[str, Any] = {
            "exclude": self._exclude_export_src_fields,
            "include": self._include_export_src_fields,
            "exclude_unset": self._exclude_unset,
            "exclude_none": self._exclude_none,
            "by_alias": not self._export_DB_by_alias,
        }
        params = self._export_helper(params=params, fields=fields, **kwargs)
        return self.model_dump_json(**params)

    async def save_json(self, filename: Path | str) -> int:
        """Save object JSON into a file"""
        filename = str2path(filename)

        try:
            if not filename.name.endswith(".json"):
                filename = filename.with_suffix(".json")
            async with open(filename, mode="w", encoding="utf-8") as rf:
                return await rf.write(self.json_src())
        except Exception as err:
            error(f"Error writing file {filename}: {err}")
        return -1

    def update(self, new: Self, match_index: bool = True) -> bool:
        """
        update instance with new. Ignore default values.
        By default matches only instance with the same index.
        """
        if match_index and self.index != new.index:
            debug(
                f"new instance has different index: {str(self.index)} != {str(new.index)}"
            )
            return False
        updated: bool = False
        for key in new.model_fields_set:
            value = getattr(new, key)
            if isinstance(value, JSONExportable):
                old = getattr(self, key)
                if isinstance(old, JSONExportable):
                    if old.update(value, match_index=match_index):
                        updated = True
                    continue
            self._set_skip_validation(key, value)
            updated = True

        return updated
