########################################################
#
# JSONExportable()
#
########################################################

import logging
from pathlib import Path
from typing import (
    AsyncGenerator,
    AsyncIterable,
    Optional,
    Type,
    Any,
    Dict,
    Tuple,
    Self,
    Literal,
    TypeVar,
    ClassVar,
    Union,
    Generic,
    Callable,
    Sequence,
)
from enum import StrEnum
from collections.abc import ItemsView, ValuesView, KeysView
from collections.abc import MutableMapping
from os import linesep
from pydantic import (
    BaseModel,
    RootModel,
    ValidationError,
    ConfigDict,
    Field,
)
import aiofiles

# from deprecated import deprecated
from .pyobjectid import PyObjectId
from .utils import str2path

TypeExcludeDict = MutableMapping[int | str, Any]

TEXT: Literal["text"] = "text"
Idx = Union[int, PyObjectId, str]
IdxType = TypeVar("IdxType", bound=Idx)
JSONExportableType = TypeVar("JSONExportableType", bound="JSONExportable")

# Setup logging
logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


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
        revalidate_instances="always",
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
    )

    # This is set in every subclass using __init_subclass__()
    _transformations: ClassVar[
        MutableMapping[Type, Callable[[Any], Optional[Self]]]
    ] = dict()

    def _set_skip_validation(self, name: str, value: Any) -> None:
        """Set fields without validation. Required to avoid
        validation loops when updating nested JSONExportables"""
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
        """Register a transformation"""
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
            error(f"failed to transform {type(in_obj)} to {cls}")
            debug(f"{err}")
        return None

    @classmethod
    def transform_many(cls, in_objs: Sequence[Any]) -> list[Self]:
        """Transform a Sequence of objects into list of Self"""
        return [out for obj in in_objs if (out := cls.transform(obj)) is not None]

    @classmethod
    def from_obj(
        cls, obj: Any, in_type: type[BaseModel] | None = None, exceptions: bool = False
    ) -> Optional[Self]:
        """Parse instance from raw object.
        Returns None if reading from object failed.
        """
        obj_in: BaseModel
        if in_type is None:
            try:
                return cls.model_validate(obj)
            except ValidationError as err:
                if exceptions:
                    raise
                error(
                    "could not parse object (%s) as %s: %s",
                    type(obj),
                    cls.__name__,
                    err,
                )
        else:
            try:
                if (obj_in := in_type.model_validate(obj)) is not None:
                    return cls.transform(obj_in)
            except ValidationError as err:
                error(
                    "could not parse object (%s) as %s: %s",
                    in_type.__name__,
                    cls.__name__,
                    err,
                )
                debug("%s", err)
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
            if (out := cls.from_obj(obj, in_type=in_type, exceptions=False)) is not None
        ]

    @classmethod
    def parse_str(cls, content: str, exceptions: bool = False) -> Self | None:
        """return class instance from a JSON string"""
        try:
            return cls.model_validate_json(content)
        except ValidationError as err:
            if exceptions:
                raise
            debug("Could not parse %s from JSON: %s", str(type(cls)), err)
        return None

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
        else:
            if "exclude" in kwargs:
                try:
                    params["exclude"].update(kwargs["exclude"])
                    del kwargs["exclude"]
                except Exception as err:
                    debug(f"'exclude' caused an error: {err}")
            if "include" in kwargs:
                try:
                    params["include"].update(kwargs["include"])
                    del kwargs["include"]
                except Exception as err:
                    debug(f"'include' caused an error: {err}")
        params.update(kwargs)
        return params

    @property
    def index(self) -> Idx:
        """return backend index"""
        raise NotImplementedError

    def flatten(self, sep: str = ".", by_alias: bool = False) -> dict[str, Any]:
        """
        return flattened representation of the object
        """

        def _flatten(obj: Any, parent_key: str = "") -> dict[str, Any]:
            flattened: dict[str, Any] = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
                    flattened.update(_flatten(value, new_key))
            elif isinstance(obj, list):
                for index, value in enumerate(obj):
                    new_key = (
                        f"{parent_key}[]{sep}{index}" if parent_key else str(index)
                    )
                    flattened.update(_flatten(value, new_key))
            elif isinstance(obj, tuple):
                for index, value in enumerate(obj):
                    new_key = (
                        f"{parent_key}(){sep}{index}" if parent_key else str(index)
                    )
                    flattened.update(_flatten(value, new_key))
            elif isinstance(obj, set):
                for index, value in enumerate(obj):
                    new_key = (
                        f"{parent_key}{{}}{sep}{index}" if parent_key else str(index)
                    )
                    flattened.update(_flatten(value, new_key))
            else:
                flattened[parent_key] = obj
            return flattened

        return _flatten(self.model_dump(by_alias=by_alias))

    @classmethod
    def from_flattened(
        cls,
        flat_dict: dict[str, Any],
        by_alias: bool = False,
        sep: str = ".",
        from_str: bool = False,
    ) -> Self:
        """
        return unflattened representation of the object
        """

        def _parse_tree(flat_dict: dict[str, Any]) -> dict[str, Any]:
            tree: dict[str, Any] = dict()
            for key, value in flat_dict.items():
                parts: list[str] = key.split(sep)
                item = tree  # dicts are assigned by reference, so this modifies the tree in place
                for i in range(len(parts) - 1):
                    if parts[i] not in item:
                        item[parts[i]] = {}
                    item = item[parts[i]]
                item[parts[-1]] = value
            return tree

        class ContainerEnum(StrEnum):
            """Helper class to identify container types in the unflattening process"""

            list = "[]"
            tuple = "()"
            set = "{}"

        def _unflatten(
            tree: dict[str, Any] | Any,
            containerType: ContainerEnum | None = None,
        ) -> Any:
            """
            Recursively unflatten a tree. If a dict has only digit keys, convert it to a list.
            If a value ("tree") is not a dict, return it as is. Assumes that the tree is nerver a list.
            The function cannot handle sets.
            """
            if isinstance(tree, dict):
                if containerType is not None:
                    if containerType == ContainerEnum.list:
                        return [_unflatten(item) for item in tree.values()]
                    elif containerType == ContainerEnum.tuple:
                        return tuple(_unflatten(item) for item in tree.values())
                    elif containerType == ContainerEnum.set:
                        return set(_unflatten(item) for item in tree.values())
                for key, value in tree.items():
                    if key.endswith("[]"):
                        tree[key[:-2]] = _unflatten(
                            value, containerType=ContainerEnum.list
                        )
                        del tree[key]
                    elif key.endswith("()"):
                        tree[key[:-2]] = _unflatten(
                            value, containerType=ContainerEnum.tuple
                        )
                        del tree[key]
                    elif key.endswith("{}"):
                        tree[key[:-2]] = _unflatten(
                            value, containerType=ContainerEnum.set
                        )
                        del tree[key]
                    else:
                        tree[key] = _unflatten(value)
            return tree

        tree: dict[str, Any] = _parse_tree(flat_dict)
        keys: list[str] = list(tree.keys())
        for key in keys:
            value = tree[key]
            if key.endswith("[]"):
                tree[key[:-2]] = _unflatten(value, containerType=ContainerEnum.list)
                del tree[key]
            elif key.endswith("()"):
                tree[key[:-2]] = _unflatten(value, containerType=ContainerEnum.tuple)
                del tree[key]
            elif key.endswith("{}"):
                tree[key[:-2]] = _unflatten(value, containerType=ContainerEnum.set)
                del tree[key]
            tree[key] = _unflatten(value)
        if from_str:
            return cls.model_validate_strings(
                tree, by_alias=by_alias, by_name=not by_alias
            )
        else:
            return cls.model_validate(tree, by_alias=by_alias, by_name=not by_alias)

    def __hash__(self) -> int:
        """
        Make object hashable using index field if defined.
        Otherwise use whole object JSON representation.
        """
        try:
            return hash(self.index)
        except NotImplementedError:
            return hash(id(self))

    def __eq__(self, value: Any) -> bool:
        if type(value) is type(self):
            return self.model_dump() == value.model_dump()
        return False

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

    @classmethod
    async def aopen_json(
        cls, filename: Path | str, exceptions: bool = False
    ) -> Self | None:
        """Load a model from a JSON file using aiofiles.open().

        Returns None on read/parse/validation errors unless exceptions=True.
        """
        try:
            async with aiofiles.open(filename, "r") as f:
                return cls.model_validate_json(await f.read())
        except ValidationError as err:
            if exceptions:
                raise
            debug("Could not parse %s from file: %s: %s", type(cls), filename, err)
        except OSError as err:
            debug("Error reading file: %s: %s", filename, err)
            if exceptions:
                raise
        return None

    @classmethod
    def open_json(cls, filename: Path | str, exceptions: bool = False) -> Self | None:
        """Load a model from a JSON file python open().

        Returns None on read/parse/validation errors unless exceptions=True.
        """
        try:
            with open(filename, "r") as f:
                return cls.model_validate_json(f.read())
        except ValidationError as err:
            if exceptions:
                raise
            debug("Could not parse %s from file: %s: %s", type(cls), filename, err)
        except OSError as err:
            debug("Error reading file: %s: %s", filename, err)
            if exceptions:
                raise
        return None

    @classmethod
    async def aimport_json(
        cls, filename: Path | str, exceptions: bool = False, **kwargs
    ) -> AsyncGenerator[Self, None]:
        """Import models from filename, one model per line"""
        try:
            # importable : JSONImportableSelf | None
            async with aiofiles.open(filename, "r") as f:
                async for line in f:
                    try:
                        if (
                            importable := cls.parse_str(
                                line, exceptions=exceptions, **kwargs
                            )
                        ) is not None:
                            yield importable
                        else:
                            error(
                                "Could not parse %s from file: %s",
                                type(cls),
                                str(filename),
                            )
                    except ValidationError:
                        if exceptions:
                            raise
        except OSError as err:
            if exceptions:
                raise
            error(f"Error importing file {filename}: {err}")

    async def asave_json(self, filename: Path | str, exceptions: bool = False) -> int:
        """
        Save an object into a JSON file
        Uses asyncio / aiofiles.open()
        """
        filename = str2path(filename)

        try:
            if not filename.name.endswith(".json"):
                filename = filename.with_suffix(".json")
            async with aiofiles.open(filename, mode="w", encoding="utf-8") as rf:
                return await rf.write(self.json_src())
        except Exception as err:
            if exceptions:
                raise
            error(f"Error writing file {filename}: {err}")
        return -1

    def save_json(self, filename: Path | str, exceptions: bool = False) -> int:
        """
        Save an object into a JSON file
        """
        filename = str2path(filename)

        try:
            if not filename.name.endswith(".json"):
                filename = filename.with_suffix(".json")
            with open(filename, mode="w", encoding="utf-8") as rf:
                return rf.write(self.json_src())
        except Exception as err:
            if exceptions:
                raise
            error(f"Error writing file {filename}: {err}")
        return -1


class JSONExportableRootDict(
    RootModel[Dict[IdxType, JSONExportableType]],
    # JSONExportable,
    Generic[IdxType, JSONExportableType],
):
    """Pydantic RootModel baseclass for JSONExportable"""

    root: Dict[IdxType, JSONExportableType] = Field(default_factory=dict)

    _sorted: bool = True  # sort items

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        from_attributes=True,
        revalidate_instances="always",
    )

    def add(self, item: JSONExportableType) -> None:
        self.root[item.index] = item  # type: ignore

    def __setitem__(self, key: IdxType, item: JSONExportableType) -> None:
        """Implement setter"""
        self.root[item.index] = item  # type: ignore

    def __getitem__(self, key: IdxType) -> JSONExportableType:
        """Implement getter"""
        return self.root[key]

    def __delitem__(self, key: IdxType) -> None:
        """Delete item with key"""
        del self.root[key]

    def __len__(self) -> int:
        """Return the number of items"""
        return len(self.root)

    def __iter__(self):
        if self._sorted:
            return iter([key for key in sorted(self.keys())])
        else:
            return iter([key for key in self.keys()])

    def values(self) -> ValuesView[JSONExportableType]:
        return self.root.values()

    def keys(self) -> KeysView[IdxType]:
        return self.root.keys()

    def __contains__(self, item: JSONExportableType | IdxType) -> bool:
        if isinstance(item, JSONExportable):
            return item.index in self.root
        else:
            return item in self.root

    def items(self) -> ItemsView[IdxType, JSONExportableType]:
        """Provide dict like functionality"""
        return self.root.items()

    def update(
        self, new: Self, match_index: bool = True
    ) -> Tuple[set[IdxType], set[IdxType]]:
        """
        update items from with 'new'. Ignore default values.
        By default matches only instance with the same index.
        """
        new_ids: set[IdxType] = {key for key in new}
        old_ids: set[IdxType] = {key for key in self}
        added: set[IdxType] = new_ids - old_ids
        updated: set[IdxType] = new_ids & old_ids

        updated = {key for key in updated if new[key] != self[key]}
        updated_idx: set[IdxType] = set()
        for key in updated:
            self[key].update(new=new[key], match_index=match_index)
            updated_idx.add(key)

        for key in added:
            self.root[key] = new[key]

        return (added, updated_idx)

    def json_src(self, **kwargs) -> str:
        """ """
        return (
            "{"
            + ",\n".join(
                [
                    f'"{str(key)}": {value.json_src(**kwargs)}'
                    for key, value in self.items()
                ]
            )
            + "}"
        )

    def json_db(self, **kwargs) -> str:
        """ """
        return (
            "{"
            + ",\n".join(
                [
                    f'"{str(key)}": {value.json_db(**kwargs)}'
                    for key, value in self.items()
                ]
            )
            + "}"
        )

    def obj_db(self, **kwargs) -> Dict[Idx, Any]:
        res: Dict[Idx, Any] = dict()
        for key, value in self.items():
            # if isinstance(key, ObjectId):
            #     key = str(key)
            res[key] = value.obj_db(**kwargs)
        return res

    def obj_src(self, **kwargs) -> Dict[Idx, Any]:
        res: Dict[Idx, Any] = dict()
        for key, value in self.items():
            # if isinstance(key, ObjectId):
            # key = str(key)
            res[key] = value.obj_src(**kwargs)
        return res

    @classmethod
    def from_obj(cls, obj: Any, exceptions: bool = False) -> Optional[Self]:
        """Parse instance from raw object.
        Returns None if reading from object failed.
        """
        try:
            return cls.model_validate(obj)
        except ValidationError as err:
            if exceptions:
                raise
            error("could not parse object as %s: %s", cls.__name__, err)
        return None

    @classmethod
    def parse_str(cls, content: str, exceptions: bool = False) -> Self | None:
        """return class instance from a JSON string"""
        try:
            return cls.model_validate_json(content, strict=True)
        except ValueError as err:
            debug(f"Could not parse {type(cls)} from JSON: {err}")
        return None

    async def asave_json(self, filename: Path | str, exceptions: bool = False) -> int:
        """
        Save object as JSON into a file using aiofiles.open()
        Uses asyncio / aiofiles.open()
        """
        filename = str2path(filename)

        try:
            if not filename.name.endswith(".json"):
                filename = filename.with_suffix(".json")
            async with aiofiles.open(filename, mode="w", encoding="utf-8") as f:
                return await f.write(self.json_src())
        except Exception as err:
            if exceptions:
                raise
            error(f"Error writing file {filename}: {err}")
        return -1

    @classmethod
    async def aopen_json(
        cls, filename: Path | str, exceptions: bool = False
    ) -> Self | None:
        """
        Open a JSON file and a return class instance
        Uses asyncio / aiofiles.open()
        Returns None if opening the JSON file fails or raises an Exception if exceptions=True
        """
        try:
            async with aiofiles.open(filename, "r") as f:
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


#############################################################
#
#
#
#############################################################


async def export_json(
    iterable: AsyncIterable[JSONExportable],
    filename: Path | str,
    force: bool = False,
    append: bool = False,
    exceptions: bool = False,
) -> tuple[int, int]:
    """
    Export data to a JSON file as one object per line.
    Uses asyncio / aiofiles.open()

    Returns number of exported / failed
    """
    # assert type(filename) is str and len(filename) > 0, "filename has to be str"
    exported: int = 0
    errors: int = 0
    try:
        exportable: JSONExportable
        if isinstance(filename, str) and filename == "-":  # STDOUT
            async for exportable in iterable:
                try:
                    print(exportable.json_src(indent=4))
                    exported += 1
                except Exception as err:
                    if exceptions:
                        raise
                    error(f"error exporting JSON type={type(exportable)}: {err}")
                    errors += 1
        else:  # FILE
            filename = str2path(filename, ".json")
            if filename.is_file() and (not (force or append)):
                raise FileExistsError(f"Cannot export to {filename}")
            mode: Literal["w", "a"] = "a" if append else "w"

            debug("opening %s for writing in mode=%s", str(filename), mode)
            async with aiofiles.open(filename, mode=mode) as txtfile:
                async for exportable in iterable:
                    try:
                        debug("writing JSON: %s", exportable.json_src())
                        await txtfile.write(exportable.json_src() + linesep)
                        exported += 1
                    except Exception as err:
                        if exceptions:
                            raise
                        errors += 1
                        error(f"Failed to export: {err}")

    except Exception as err:
        if exceptions:
            raise
        error(f"Failed exporting to JSON: {err}")
    return exported, errors
