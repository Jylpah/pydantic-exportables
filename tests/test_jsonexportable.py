import pytest  # type: ignore
from typing import Self, List, Annotated, TypeVar
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from datetime import date, datetime
from enum import StrEnum, IntEnum
from bson import ObjectId
import json
import logging
import aiofiles

# from unittest.mock import patch
from pydantic_exportables import (
    JSONExportable,
    JSONExportableRootDict,
    Idx,
    AliasMapper,
    PyObjectId,
    epoch_now,
    str2path,
)

logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


##########################################################
#
# Helper functions
#
###########################################################

B = TypeVar("B", bound=BaseModel)


async def open_json(
    model: type[B], filename: Path | str, exceptions: bool = False
) -> B | None:
    """Open replay JSON file and return class instance"""
    try:
        async with aiofiles.open(filename, "r") as f:
            return model.model_validate_json(await f.read())
    except ValueError as err:
        debug(f"Could not parse {type(model)} from file: {filename}: {err}")
        if exceptions:
            raise
    except OSError as err:
        debug(f"Error reading file: {filename}: {err}")
        if exceptions:
            raise
    return None


async def save_json(
    obj: JSONExportable | JSONExportableRootDict, filename: Path | str
) -> int:
    """Save object JSON into a file"""
    filename = str2path(filename)
    #
    try:
        if not filename.name.endswith(".json"):
            filename = filename.with_suffix(".json")
        async with aiofiles.open(filename, mode="w", encoding="utf-8") as rf:
            return await rf.write(obj.json_src())
    except Exception as err:
        error(f"Error writing file {filename}: {err}")
    return -1


class FakeAsyncFile:
    def __init__(self, content: str = "") -> None:
        self.content = content

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def write(self, data: str) -> int:
        self.content += data
        return len(data)


class FakeAsyncOpen:
    def __init__(self, content: str = "") -> None:
        self.file = FakeAsyncFile(content)

    def __call__(self, filename: Path, mode: str = "r", **kwargs) -> FakeAsyncFile:
        if mode == "w":
            self.file.content = ""
        return self.file


class Eyes(StrEnum):
    blue = "Blue"
    grey = "Grey"
    brown = "Brown"


class Hair(IntEnum):
    black = 0
    brown = 1
    red = 2
    blonde = 3


class JSONChild(JSONExportable):
    name: str
    born: int = Field(default_factory=epoch_now)

    @property
    def index(self) -> Idx:
        """return backend index"""
        return self.name


class JSONParent(JSONExportable):
    name: str = Field(alias="n")
    years: int = Field(default=37, alias="y")
    married: bool = Field(default=True, alias="m")
    array: List[str] = Field(default_factory=list, alias="a")
    child: JSONChild | None = Field(default=None, alias="c")

    _exclude_unset = False

    @property
    def index(self) -> Idx:
        """return backend index"""
        return self.name


class JSONAdult(JSONExportable):
    name: str
    age: int = Field(default=40)
    married: Annotated[bool, Field(default=False)] = False

    model_config = ConfigDict(extra="forbid")

    def transform2JSONParent(self) -> JSONParent:
        return JSONParent(
            name=self.name, years=self.age, married=self.married, child=None
        )


class JSONNeighbours(JSONExportableRootDict[str, JSONParent]):
    pass


JSONParent.register_transformation(JSONAdult, JSONAdult.transform2JSONParent)


def datetime_as_str() -> str:
    return datetime.now().isoformat()


class JSONIntIdxExportable(JSONExportable):
    idx: int = Field(default=...)
    name: str = Field(default_factory=datetime_as_str)

    @property
    def index(self) -> Idx:
        return self.idx


class IntIdxExportableDict(JSONExportableRootDict[int, JSONIntIdxExportable]):
    pass


class ObjectIdExportable(JSONExportable):
    id: PyObjectId = Field(default_factory=PyObjectId)
    name: str = Field(default=..., alias="n")

    @property
    def index(self) -> Idx:
        return self.id


class ObjectIdExportableDict(JSONExportableRootDict[PyObjectId, ObjectIdExportable]):
    pass


def today() -> datetime:
    return datetime.combine(date.today(), datetime.min.time())


def str2datetime(dt: str) -> datetime:
    debug("str2datetime(): %s", dt)
    return datetime.combine(date.fromisoformat(dt), datetime.min.time())


def datetime2str(dt: datetime) -> str:
    debug("datetime2str(): %s", dt)
    return dt.date().isoformat()


# class TXTPerson(TXTExportable, TXTImportable, CSVExportable, Importable):
#     name: str = Field(default=...)
#     age: int = Field(default=...)
#     height: float = Field(default=...)
#     birthday: datetime = Field(default_factory=today)
#     woman: bool = Field(default=False)
#     hair: Hair = Field(default=Hair.brown)
#     eyes: Eyes = Field(default=Eyes.blue)

#     _csv_custom_readers = {"birthday": str2datetime}
#     _csv_custom_writers = {"birthday": datetime2str}

#     def txt_row(self, format: str = "") -> str:
#         """export data as single row of text"""
#         return f"{self.name}:{self.age}:{self.height}:{self.birthday.date().isoformat()}:{self.woman}:{self.hair.name}:{self.eyes.name}"

#     @classmethod
#     def from_txt(cls, text: str, **kwargs) -> Self:
#         """Provide parse object from a line of text"""
#         debug(f"line: {text}")
#         n, a, h, bd, w, ha, e = text.split(":")
#         debug(
#             "name=%s, age=%s height=%s, birthday=%s, woman=%s, hair=%s, eyes=%s",
#             n,
#             a,
#             h,
#             bd,
#             w,
#             ha,
#             e,
#         )
#         return cls(
#             name=n,
#             age=int(a),
#             height=float(h),
#             birthday=datetime.fromisoformat(bd),
#             woman=(w == "True"),
#             hair=Hair[ha],
#             eyes=Eyes[e],
#             **kwargs,
#         )

#     def __hash__(self) -> int:
#         """Make object hashable, but using index fields only"""
#         return hash((self.name, self.birthday.date()))


def rm_parenthesis(name: str) -> str:
    return name.removesuffix("()")


def add_parenthesis(name: str) -> str:
    return f"{name}()"


# class CSVPerson(TXTPerson):
#     favorite_func: str

#     _csv_custom_readers = {"favorite_func": add_parenthesis}
#     _csv_custom_writers = {"favorite_func": rm_parenthesis}


# class CSVChild(CSVPerson):
#     favorite_func: str

#     _csv_custom_readers = {"toy": str.lower}
#     _csv_custom_writers = {"toy": str.upper}


@pytest.fixture
def json_parents() -> List[JSONParent]:
    c1 = JSONChild(name="c1")
    c3 = JSONChild(name="c3")
    res: List[JSONParent] = list()
    res.append(JSONParent(name="Erik", years=1, array=["one", "two"], child=c1))
    res.append(JSONParent(name="Mia", years=-6, array=["three", "four"]))
    res.append(JSONParent(name="Jack", years=-6, child=c3))
    return res


@pytest.fixture
def json_adults() -> List[JSONAdult]:
    res: List[JSONAdult] = list()
    res.append(JSONAdult(name="Alice", age=35))
    res.append(JSONAdult(name="Bob", age=38))
    return res


# @pytest.fixture
# def csv_data() -> List[CSVPerson]:
#     res: List[CSVPerson] = list()
#     res.append(
#         CSVPerson(
#             name="Marie",
#             age=0,
#             height=1.85,
#             woman=True,
#             eyes=Eyes.brown,
#             hair=Hair.red,
#             favorite_func="VLOOKUP()",
#         )
#     )
#     res.append(
#         CSVPerson(
#             name="Jack Who",
#             age=45,
#             height=1.43,
#             birthday=datetime.fromisoformat("1977-07-23"),
#             eyes=Eyes.grey,
#             favorite_func="INDEX()",
#         )
#     )
#     res.append(
#         CSVPerson(
#             name="James 3.5",
#             age=18,
#             height=1.76,
#             birthday=datetime.fromisoformat("2005-02-14"),
#             hair=Hair.blonde,
#             favorite_func="SUMPRODUCT()",
#         )
#     )
#     return res


# @pytest.fixture
# def txt_data() -> List[TXTPerson]:
#     res: List[TXTPerson] = list()
#     res.append(
#         TXTPerson(
#             name="Marie", age=0, height=1.85, woman=True, eyes=Eyes.brown, hair=Hair.red
#         )
#     )
#     res.append(
#         TXTPerson(
#             name="Jack Who",
#             age=45,
#             height=1.43,
#             birthday=datetime.fromisoformat("1977-07-23"),
#             eyes=Eyes.grey,
#         )
#     )
#     res.append(
#         TXTPerson(
#             name="James 3.5",
#             age=18,
#             height=1.76,
#             birthday=datetime.fromisoformat("2005-02-14"),
#             hair=Hair.blonde,
#         )
#     )
#     return res


# @pytest.mark.asyncio
# async def test_1_json_exportable_import_save(
#     tmp_path: Path, json_parents: List[JSONParent]
# ):
#     fn: Path = tmp_path / "export.json"

#     await export(awrap(json_parents), format="json", filename="-")  # type: ignore
#     await export(awrap(json_parents), format="json", filename=fn)  # type: ignore
#     await export(
#         awrap(json_parents), format="json", filename=str(fn.resolve()), force=True
#     )  # type: ignore

#     imported: set[JSONParent] = set()
#     try:
#         async for p_in in JSONParent.import_file(fn):
#             imported.add(p_in)
#     except Exception as err:
#         assert False, f"failed to import test data: {err}"

#     for data in json_parents:
#         try:
#             imported.remove(data)
#         except Exception as err:
#             assert False, f"could not export or import item: {data}: {err}"

#     assert len(imported) == 0, "Export or import failed"

#     for parent in json_parents:
#         assert await parent.save_json(fn) > 0, (
#             f"could not save JSONExportable: {str(parent)}"
#         )
#         assert (parent_imported := await JSONParent.open_json(fn)) is not None, (
#             f"could not import save json: {str(parent)}"
#         )
#         assert parent == parent_imported, (
#             f"imported data does not match original: original={parent}, imported={parent_imported}"
#         )
#         assert (adult := await JSONAdult.open_json(fn)) is None, (
#             f"open_json() returned instance even it should not: {adult}"
#         )

#     parent = json_parents[0]
#     assert await parent.save_json(fn.with_suffix("")) > 0, (
#         f"could not save JSONExportable: {str(parent)}"
#     )

#     # Test for opening non-existent file
#     assert (
#         adult := await JSONAdult.open_json(fn.with_suffix(".not-found.json"))
#     ) is None, f"open_json() returned instance from non-existent file: {adult}"


# @pytest.mark.asyncio
# async def test_export_json_writes_jsonl_and_returns_stats(
#     tmp_path: Path, json_parents: List[JSONParent]
# ) -> None:
#     fn = tmp_path / "parents"
#     fake_open = FakeAsyncOpen()

#     with patch("pydantic_exportables.jsonexportable.open", side_effect=fake_open):
#         stats = await export_json(awrap(json_parents), fn)  # type: ignore[arg-type]

#     assert stats.get("rows") == len(json_parents), (
#         f"export_json() logged wrong row count: {stats.get('rows')}"
#     )
#     assert stats.get("errors") == 0, (
#         f"export_json() logged unexpected errors: {stats.get('errors')}"
#     )

#     lines = fake_open.file.content.splitlines()
#     assert len(lines) == len(json_parents), (
#         f"export_json() wrote wrong number of lines: {len(lines)}"
#     )

#     for line, parent in zip(lines, json_parents):
#         assert (parsed := JSONParent.parse_str(line)) is not None, (
#             f"export_json() wrote invalid JSON line: {line}"
#         )
#         assert parsed == parent, (
#             f"export_json() wrote wrong content: {parsed} != {parent}"
#         )


# @pytest.mark.asyncio
# async def test_export_json_appends_when_requested(
#     tmp_path: Path, json_parents: List[JSONParent]
# ) -> None:
#     fn = tmp_path / "append.json"
#     fake_open = FakeAsyncOpen()

#     with patch("pydantic_exportables.jsonexportable.open", side_effect=fake_open):
#         first_stats = await export_json(awrap(json_parents[:1]), fn)  # type: ignore[arg-type]
#         append_stats = await export_json(awrap(json_parents[1:]), fn, append=True)  # type: ignore[arg-type]

#     assert first_stats.get("rows") == 1, (
#         f"initial export logged wrong row count: {first_stats.get('rows')}"
#     )
#     assert append_stats.get("rows") == len(json_parents[1:]), (
#         f"append export logged wrong row count: {append_stats.get('rows')}"
#     )

#     lines = fake_open.file.content.splitlines()
#     assert len(lines) == len(json_parents), (
#         f"append export wrote wrong number of lines: {len(lines)}"
#     )

#     for line, parent in zip(lines, json_parents):
#         assert JSONParent.parse_str(line) == parent, (
#             f"append export changed row order or content: {line}"
#         )


# @pytest.mark.asyncio
# async def test_export_json_existing_file_requires_force(
#     tmp_path: Path, json_parents: List[JSONParent]
# ) -> None:
#     fn = tmp_path / "existing.json"
#     fn.write_text("old-data\n", encoding="utf-8")
#     fake_open = FakeAsyncOpen("old-data\n")

#     with pytest.raises(FileExistsError):
#         await export_json(awrap(json_parents), fn)  # type: ignore[arg-type]

#     with patch("pydantic_exportables.jsonexportable.open", side_effect=fake_open):
#         stats = await export_json(awrap(json_parents[:1]), fn, force=True)  # type: ignore[arg-type]

#     assert stats.get("rows") == 1, (
#         f"forced export logged wrong row count: {stats.get('rows')}"
#     )
#     lines = fake_open.file.content.splitlines()
#     assert lines == [json_parents[0].json_src()], (
#         "force=True did not overwrite existing file content"
#     )


# @pytest.mark.asyncio
# async def test_export_json_stdout(capsys: pytest.CaptureFixture[str]) -> None:
#     parent = JSONParent(name="Erik", years=1, array=["one", "two"])

#     stats = await export_json(awrap([parent]), "-")  # type: ignore[arg-type]

#     captured = capsys.readouterr()
#     assert captured.out.strip() == parent.json_src(indent=4), (
#         "export_json() did not print formatted JSON to stdout"
#     )
#     assert stats.get("rows") == 1, (
#         f"stdout export logged wrong row count: {stats.get('rows')}"
#     )
#     assert stats.get("errors") == 0, (
#         f"stdout export logged unexpected errors: {stats.get('errors')}"
#     )

## OK


@pytest.mark.asyncio
async def test_2_json_exportable_include_exclude() -> None:
    # test for custom include/exclude
    parent = JSONParent(
        name="Jack", years=26, married=True, child=JSONChild(name="Nick")
    )

    mapper = AliasMapper(JSONParent)
    parent_src: dict
    parent_db: dict
    parent_src = json.loads(parent.json_src())
    assert "array" in parent_src, (
        "json_src() failed: _exclude_unset set 'False', 'array' excluded"
    )
    parent_db = json.loads(parent.json_db())
    assert "m" not in parent_db, (
        "json_db() failed: _exclude_defaults set 'True', 'c' included"
    )

    for excl, incl in zip(["child", None], ["name", None]):
        kwargs: dict[str, set[str]] = dict()
        if excl is not None:
            kwargs["exclude"] = {excl}
        if incl is not None:
            kwargs["include"] = {incl}

        parent_src = json.loads(parent.json_src(fields=None, **kwargs))
        parent_db = json.loads(parent.json_db(fields=None, **kwargs))
        if excl is not None:
            assert mapper.alias(excl) not in parent_db, (
                f"json_src() failed: excluded field {excl} included"
            )
            assert excl not in parent_src, (
                f"json_db() failed: excluded field {excl} included"
            )
        if incl is not None:
            assert incl in parent_src, (
                f"json_src() failed: included field {incl} excluded"
            )
            assert mapper.alias(incl) in parent_db, (
                f"json_db() failed: included field {incl} excluded"
            )

    parent_src = parent.obj_src(fields=["name", "array"])
    assert "years" not in parent_src, (
        "json_src() failed: excluded field 'years' included"
    )
    assert "array" in parent_src, "json_src() failed: included field 'array' excluded"

    parent_db = parent.obj_db(fields=["name", "array"])
    assert mapper.alias("years") not in parent_db, (
        "json_db() failed: excluded field 'years' included"
    )
    assert mapper.alias("array") in parent_db, (
        "json_db() failed: included field 'array' excluded"
    )

    parent_src = parent.obj_src()
    assert (parent_new := JSONParent.from_obj(parent_src)) is not None, (
        "could not create object from exported model"
    )
    assert parent == parent_new, (
        f"re-created object is different to original: {parent_new}"
    )

    parent_db = parent.obj_db()
    assert (parent_new := JSONParent.from_obj(parent_db)) is not None, (
        "could not create object from exported model"
    )
    assert parent == parent_new, (
        f"re-created object is different to original: {parent_new}"
    )

    parent_fail = parent.obj_src()
    del parent_fail["name"]
    assert (_ := JSONParent.from_obj(parent_fail)) is None, (
        f"from_obj() return an instance from faulty data: {parent_fail}"
    )


def test_3_jsonexportable_update(json_parents: List[JSONParent]):
    """
    test for JSONExportable.update()
    """
    p0: JSONParent = json_parents[0]
    p1: JSONParent = json_parents[1]
    p2: JSONParent = json_parents[2]

    p: JSONParent = p0.model_copy(deep=True)

    for new in json_parents[1:]:
        assert not p.update(new, match_index=True), (
            "update succeeded even the indexes do not match"
        )
    assert p.update(p1, match_index=False), (
        "update did not succeeded even the indexes were ignored"
    )
    assert all(
        [
            p.name == p1.name,
            p.years == p1.years,
            p.married == p1.married,
            p.array == p1.array,
            p.child == p0.child,
        ]
    ), f"update() failed: updated={str(p)}"
    assert p.update(p2, match_index=False), (
        "update did not succeeded even the indexes were ignored"
    )
    assert all(
        [
            p.name == p2.name,
            p.years == p2.years,
            p.married == p2.married,
            p.array == p1.array,
            p.child == p2.child,
        ]
    ), f"update() failed: updated={str(p)}"


def test_4_jsonexportable_transform(json_adults: List[JSONAdult]):
    res = JSONParent.transform_many(json_adults)
    N: int = len(json_adults)
    assert len(res) == N, f"could not transform all data: {len(res)} != {N}"

    res2 = JSONAdult.transform_many(json_adults)
    assert len(res2) == N, f"could not transform all data: {len(res)} != {N}"

    res3 = JSONParent.from_objs(
        [adult.model_dump() for adult in json_adults], in_type=JSONAdult
    )
    assert len(res3) == N, f"from_objs(in_type=JSONAdult) failed: {len(res)} != {N}"


def test_5_jsonexportable_transform_fails(json_parents: List[JSONParent]):
    res = JSONAdult.transform_many(json_parents)
    assert len(res) == 0, f"transform data it should have not: {len(res)} != 0"

    res2 = JSONChild.from_objs(
        [parent.model_dump() for parent in json_parents], in_type=JSONAdult
    )
    assert len(res2) == 0, (
        f"from_objs() returned data when it should have not: {len(res)} != 0"
    )


def test_6_parse_str(json_parents: List[JSONParent]) -> None:
    for parent in json_parents:
        # test failure
        assert (_ := JSONAdult.parse_str(parent.json_src())) is None, (
            f"parse_str() returned instance from faulty data: {parent.json_src()}"
        )
        # test success
        json_str: str = parent.json_src()
        parsed: JSONParent | None = JSONParent.parse_str(json_str)
        assert parsed is not None, f"parse_str() failed to parse valid JSON: {json_str}"
        assert parsed == parent, (
            f"parse_str() returned different object than original: {parsed} != {parent}"
        )


def test_7_jsonexportablerootdict(json_parents: List[JSONParent]) -> None:
    family = JSONNeighbours()
    for parent in json_parents:
        family.add(parent)

    assert len(family) == len(json_parents), (
        f"could not add all the list members: {family} != {len(json_parents)}"
    )

    family2 = JSONNeighbours()
    parent = JSONParent(
        name="Erik",
        years=28,
        child=None,
    )
    family2[parent.name] = parent
    family2.add(
        JSONParent(
            name="Betty",
            years=26,
            child=JSONChild(
                name="Elisabeth",
                born=int(
                    datetime(year=2009, month=6, day=4, hour=13, minute=56).timestamp()
                ),
            ),
        )
    )
    added, updated = family.update(family2)
    assert len(added) == 1 and len(updated) == 1, "update() failed"
    assert family[parent.name].name == parent.name, "__get_item__() failed"
    if parent in family:
        pass
    if parent.name in family:
        del family[parent.name]
    else:
        assert False, f"item '{parent.name}' not found in family even it should"

    for name, parent in family.items():
        assert name == parent.name, "wrong item returned"

    assert len(family) == len(family.values()), (
        f"values() returned incorrect number of items: {len(family)} != {len(family.values())}"
    )

    assert (_ := JSONNeighbours.from_obj(family.obj_src())) is not None, (
        f"could not recreate JSONExportableRootDict() from obj_src(): {str(family.obj_src())}"
    )

    assert (_ := JSONNeighbours.from_obj(family.obj_db())) is not None, (
        f"could not recreate JSONExportableRootDict() from obj_db(): {str(family.obj_db())}"
    )

    # debug(family.json_src())
    assert (_ := JSONNeighbours.parse_str(family.json_src())) is not None, (
        f"could not parse JSONExportableRootDict() from json_src(): {family.json_src()}"
    )

    assert (_ := JSONNeighbours.parse_str(family.json_db())) is not None, (
        f"could not parse JSONExportableRootDict() from json_db(): {family.json_db()}"
    )


def test_8_export_helper_edge_cases() -> None:
    parent = JSONParent(
        name="Test",
        years=25,
        married=True,
        array=["a", "b"],
        child=JSONChild(name="Kid"),
    )

    # Test with both include and exclude in kwargs
    result: dict = parent._export_helper({"exclude": ["child"]}, exclude={"array"})
    assert "child" not in result["exclude"], (
        "exclude parameter should be passed to export_helper"
    )
    assert "array" in result["exclude"], (
        "exclude parameter should be passed to export_helper"
    )

    # Test with fields parameter
    result = parent._export_helper({"exclude": set()}, fields=["name", "years"])
    assert result["include"] == {"name": True, "years": True}
    assert not result["exclude_defaults"], (
        f"exclude_defaults should be False when fields are provided, but got {result['exclude_defaults']}"
    )
    assert not result["exclude_unset"], (
        f"exclude_unset should be False when fields are provided, but got {result['exclude_unset']}"
    )
    assert not result["exclude_none"], (
        f"exclude_none should be False when fields are provided, but got {result['exclude_none']}"
    )


def test_9_from_obj_validation_error() -> None:
    invalid_data: dict[str, str] = {"name": "Test", "years": "not_a_number"}
    assert JSONParent.from_obj(invalid_data) is None, (
        "from_obj() should return None for invalid data"
    )


def test_10_transform_exception_handling() -> None:
    # Register a transformation that raises an exception
    def failing_transform(obj):
        raise ValueError("Test error")

    JSONParent.register_transformation(str, failing_transform)
    assert JSONParent.transform("test_string") is None, "transform() failed"


def test_11_index_property(json_parents) -> None:
    for parent in json_parents:
        assert isinstance(parent.index, (str, int, PyObjectId))
        assert parent.index == parent.name  # Based on implementation


def test_12_PyObjectIdasIdx() -> None:
    d = ObjectIdExportableDict()

    L: int = 10
    for i in range(L):
        d.add(ObjectIdExportable(id=ObjectId(), name=f"Name {i}"))

    debug(d.json_src(exclude_defaults=False))
    assert len(d) == L, f"could not add all the items: {len(d)} != {L}"

    assert (imported := ObjectIdExportableDict.parse_str(d.json_src())) is not None, (
        "could not parse exported JSON"
    )
    assert len(imported) == len(d), (
        f"the size of the imported 'ObjectIdExportableDict' does not match the exporterd: {len(imported)} != {len(d)}"
    )

    for key, value in imported.items():
        assert isinstance(key, ObjectId), (
            f"the imported keys are not type of ObjectId, but {type(key)}"
        )
        assert isinstance(value.index, ObjectId), (
            f"imported objects 'id' field is {type(value.index)}, not ObjectId"
        )

    d = ObjectIdExportableDict()

    for i in range(L):
        d.add(ObjectIdExportable(name=f"Name {i}"))

    debug(d)
    assert len(d) == L, (
        f"could not add all the items with generated ObjectId: {len(d)} != {L}"
    )


@pytest.mark.asyncio
async def test_13_IntasIdx(tmp_path: Path) -> None:
    d = IntIdxExportableDict()
    export_fn: Path = tmp_path / "test_13_IntasIdx.json"
    L: int = 20
    for i in range(L):
        d.add(JSONIntIdxExportable(idx=i))

    debug(d.json_src(exclude_defaults=False))
    assert len(d) == L, f"could not add all the items: {len(d)} != {L}"
    assert await save_json(d, export_fn) > 0, (
        "could not export IntIdxExportableDict to JSON"
    )

    assert (imported := await open_json(IntIdxExportableDict, export_fn)) is not None, (
        "could not import IntIdxExportableDict from JSON"
    )

    assert len(imported) == len(d), (
        f"the imported object has incorrect number of objects: {len(imported)} != {len(d)}"
    )

    for key, value in imported.items():
        assert isinstance(key, int), f"imported key is not type 'int', but {type(key)}"
        assert isinstance(value.index, int), (
            f"imported objects 'id' field is {type(value.index)}, not int"
        )


def test_14_jsonexportable_hash():
    parent1 = JSONParent(name="Alice", years=30)
    parent2 = JSONParent(name="Alice", years=35)
    parent3 = JSONParent(name="Bob", years=30)

    # Same index should have same hash
    assert hash(parent1) == hash(parent2), (
        "__hash__() failed: same index should have same hash"
    )
    # Different index should have different hash
    assert hash(parent1) != hash(parent3), (
        "__hash__() failed: different index should have different hash"
    )

    # Can be used in sets
    parent_set = {parent1, parent2, parent3}
    assert len(parent_set) == 3, (
        "__hash__() failed: hashable objects should still be distinguished by identity when __eq__ is not overridden"
    )


def test_15_flatten_jsonexportable() -> None:
    child = JSONChild(name="Child", born=1234567890)
    parent = JSONParent(
        name="Parent",
        years=40,
        married=True,
        array=["a", "b"],
        child=child,
    )

    flat_dict = parent.flatten()
    expected_keys = {
        "name",
        "years",
        "married",
        "array[].0",
        "array[].1",
        "child.name",
        "child.born",
    }
    assert set(flat_dict.keys()) == expected_keys, (
        f"flattened_dict() returned incorrect keys: {set(flat_dict.keys())} != {expected_keys}"
    )
    assert flat_dict["name"] == parent.name, (
        "flattened_dict() returned incorrect value for 'name'"
    )
    assert flat_dict["years"] == parent.years, (
        "flattened_dict() returned incorrect value for 'years'"
    )
    assert flat_dict["married"] == parent.married, (
        "flattened_dict() returned incorrect value for 'married'"
    )
    assert flat_dict["array[].0"] == parent.array[0], (
        "flattened_dict() returned incorrect value for 'array'"
    )
    assert flat_dict["array[].1"] == parent.array[1], (
        "flattened_dict() returned incorrect value for 'array'"
    )
    assert parent.child is not None and flat_dict["child.name"] == parent.child.name, (
        "flattened_dict() returned incorrect value for 'child.name'"
    )
    assert flat_dict["child.born"] == parent.child.born, (
        "flattened_dict() returned incorrect value for 'child.born'"
    )


def test_16_jsonexportable_from_flattened():
    child = JSONChild(name="Child", born=1234567890)
    parent = JSONParent(
        name="Parent",
        years=40,
        married=True,
        array=["a", "b"],
        child=child,
    )

    assert (parent_new := JSONParent.from_flattened(parent.flatten())) is not None, (
        "from_flattened() could not recreate instance from flatten()"
    )
    assert parent_new == parent, "from_flattened() did not reverse flatten()"


def test_17_jsonexportable_from_flattened_custom_separator():
    child = JSONChild(name="Child", born=1234567890)
    parent = JSONParent(
        name="Parent",
        years=40,
        married=True,
        array=["a", "b"],
        child=child,
    )

    assert (
        parent_new := JSONParent.from_flattened(parent.flatten(sep="__"), sep="__")
    ) is not None, "from_flattened() could not recreate instance with custom separator"
    assert parent_new == parent, (
        "from_flattened() did not reverse flatten() with custom separator"
    )


def test_18_jsonexportable_from_flattened_numeric_dict_keys() -> None:
    class JSONNumericDict(JSONExportable):
        name: str
        scores: dict[int, str]

        @property
        def index(self) -> Idx:
            return self.name

    obj = JSONNumericDict(name="numbers", scores={0: "zero", 1: "one"})

    assert (obj_new := JSONNumericDict.from_flattened(obj.flatten())) is not None, (
        "from_flattened() could not recreate instance with numeric dict keys"
    )
    assert obj_new == obj, "from_flattened() converted numeric dict keys incorrectly"


def test_19_jsonexportable_from_flattened(json_parents: List[JSONParent]) -> None:
    for parent in json_parents:
        flat_dict = parent.flatten()
        assert isinstance(flat_dict, dict), "flatten() should return a dictionary"
        assert len(flat_dict) > 0, "flatten() returned an empty dictionary"
        assert parent == JSONParent.from_flattened(flat_dict), (
            "from_flattened() did not reverse flatten() correctly"
        )


# @pytest.mark.asyncio
# async def test_15_import_json(tmp_path: Path, json_parents: list[JSONParent]):
#     fn: Path = tmp_path / "multi.json"

#     # Write multiple JSON lines
#     content: str = "\n".join(parent.json_src() for parent in json_parents)
#     with open(fn, "w") as f:
#         f.write(content)

#     # Import and verify
#     imported: list[JSONParent] = []
#     async for item in JSONParent.import_json(fn):
#         imported.append(item)

#     assert len(imported) == len(json_parents)
#     for original, imported_item in zip(json_parents, imported):
#         assert original == imported_item


# @pytest.mark.asyncio
# async def test_16_async_error_handling(tmp_path: Path, json_parents: List[JSONParent]):
#     # Test open_json with invalid JSON
#     invalid_fn: Path = tmp_path / "invalid.json"
#     with open(invalid_fn, "w") as f:
#         f.write("invalid json content")

#     assert await JSONParent.open_json(invalid_fn) is None, (
#         "open_json() should return None for invalid JSON content"
#     )

#     # Test import_json with invalid lines
#     mixed_fn: Path = tmp_path / "mixed.json"
#     valid_json: str = json_parents[0].json_src()
#     with open(mixed_fn, "w") as f:
#         f.write(f"{valid_json}\ninvalid line\n{valid_json}")

#     imported: list[JSONParent] = []
#     async for item in JSONParent.import_json(mixed_fn):
#         imported.append(item)

#     assert len(imported) == 2, (
#         f"Expected 2 valid items, but got {len(imported)}"
#     )  # Only valid lines imported


# @pytest.mark.asyncio
# async def test_17_jsonexportablerootdict_save_open_json(tmp_path, json_parents):
#     family = JSONNeighbours()
#     for parent in json_parents:
#         family.add(parent)

#     fn = tmp_path / "family.json"

#     # Test save_json
#     assert await family.save_json(fn) > 0

#     # Test open_json
#     loaded = await JSONNeighbours.open_json(fn)
#     assert loaded is not None
#     assert len(loaded) == len(family)
#     for key in family:
#         assert loaded[key] == family[key]


# @pytest.mark.asyncio
# async def test_18_txt_exportable_importable(tmp_path: Path, txt_data: List[TXTPerson]):
#     fn: Path = tmp_path / "export.txt"

#     await export(awrap(txt_data), "txt", filename="-")  # type: ignore
#     await export(awrap(txt_data), "txt", filename=fn)  # type: ignore
#     await export(awrap(txt_data), format="txt", filename=str(fn.resolve()), force=True)  # type: ignore

#     imported: set[TXTPerson] = set()
#     try:
#         async for p_in in TXTPerson.import_file(fn):
#             debug("import_txt(): %s", str(p_in))
#             imported.add(p_in)
#     except Exception as err:
#         assert False, f"failed to import test data: {err}"

#     assert len(imported) == len(txt_data), (
#         f"failed to import all data from TXT file: {len(imported)} != {len(txt_data)}"
#     )

#     for data in txt_data:
#         debug("trying to remove data=%s from imported", str(data))
#         try:
#             imported.remove(data)
#         except Exception as err:
#             assert False, f"could not export or import item: {data}: {err}: {imported}"

#     assert len(imported) == 0, "Export or import failed"


# @pytest.mark.asyncio
# async def test_19_csv_exportable_importable(tmp_path: Path, csv_data: List[CSVPerson]):
#     fn: Path = tmp_path / "export.csv"

#     await export(awrap(csv_data), "csv", filename="-")  # type: ignore
#     await export(awrap(csv_data), "csv", filename=fn)  # type: ignore
#     await export(awrap(csv_data), "csv", filename=str(fn.resolve()), force=True)  # type: ignore

#     imported: set[CSVPerson] = set()
#     try:
#         async for p_in in CSVPerson.import_file(fn):
#             debug("imported: %s", str(p_in))
#             imported.add(p_in)
#     except Exception as err:
#         assert False, f"failed to import test data: {err}"

#     assert len(imported) == len(csv_data), (
#         f"could not import all CSV data: {len(imported)} != {len(csv_data)}"
#     )
#     for data_imported in imported:
#         debug("hash(data_imported)=%d", hash(data_imported))
#         try:
#             if data_imported in csv_data:
#                 ndx: int = csv_data.index(data_imported)
#                 data = csv_data[ndx]
#                 assert data == data_imported, (
#                     f"imported data different from original: {data_imported} != {data}"
#                 )
#                 csv_data.pop(ndx)
#             else:
#                 assert False, f"imported data not in the original: {data_imported}"
#         except ValueError:
#             assert False, (
#                 f"export/import conversion error. imported data={data_imported} is not in input data"
#             )

#     assert len(csv_data) == 0, (
#         f"could not import all the data correctly: {len(csv_data)} != 0"
#     )
