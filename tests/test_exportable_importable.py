import pytest  # type: ignore
from typing import Self, List, Annotated
from pydantic import Field, ConfigDict
from pathlib import Path
from datetime import date, datetime
from enum import StrEnum, IntEnum

from bson import ObjectId
import json
import logging
from pydantic_exportables import (
    JSONExportable,
    JSONExportableRootDict,
    export,
    Idx,
    CSVExportable,
    TXTExportable,
    TXTImportable,
    Importable,
    AliasMapper,
    PyObjectId,
    awrap,
    epoch_now,
)

########################################################
#
# Test Plan
#
########################################################

# 1) Create instances, export as JSON, import back and compare
# 2) Create instances, export as CSV, import back and compare
# 3) Create instances, export as TXT, import back and compare

logger = logging.getLogger(__name__)
error = logger.error
message = logger.warning
verbose = logger.info
debug = logger.debug


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

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        return {"name": self.index}


class JSONParent(JSONExportable, Importable):
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

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        return {"name": self.index}


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
    id: Annotated[PyObjectId, Field(default_factory=ObjectId)] = PyObjectId()
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


class TXTPerson(TXTExportable, TXTImportable, CSVExportable, Importable):
    name: str = Field(default=...)
    age: int = Field(default=...)
    height: float = Field(default=...)
    birthday: datetime = Field(default_factory=today)
    woman: bool = Field(default=False)
    hair: Hair = Field(default=Hair.brown)
    eyes: Eyes = Field(default=Eyes.blue)

    _csv_custom_readers = {"birthday": str2datetime}
    _csv_custom_writers = {"birthday": datetime2str}

    def txt_row(self, format: str = "") -> str:
        """export data as single row of text"""
        return f"{self.name}:{self.age}:{self.height}:{self.birthday.date().isoformat()}:{self.woman}:{self.hair.name}:{self.eyes.name}"

    @classmethod
    def from_txt(cls, text: str, **kwargs) -> Self:
        """Provide parse object from a line of text"""
        debug(f"line: {text}")
        n, a, h, bd, w, ha, e = text.split(":")
        debug(
            "name=%s, age=%s height=%s, birthday=%s, woman=%s, hair=%s, eyes=%s",
            n,
            a,
            h,
            bd,
            w,
            ha,
            e,
        )
        return cls(
            name=n,
            age=int(a),
            height=float(h),
            birthday=datetime.fromisoformat(bd),
            woman=(w == "True"),
            hair=Hair[ha],
            eyes=Eyes[e],
            **kwargs,
        )

    def __hash__(self) -> int:
        """Make object hashable, but using index fields only"""
        return hash((self.name, self.birthday.date()))


def rm_parenthesis(name: str) -> str:
    return name.removesuffix("()")


def add_parenthesis(name: str) -> str:
    return f"{name}()"


class CSVPerson(TXTPerson):
    favorite_func: str

    _csv_custom_readers = {"favorite_func": add_parenthesis}
    _csv_custom_writers = {"favorite_func": rm_parenthesis}


class CSVChild(CSVPerson):
    favorite_func: str

    _csv_custom_readers = {"toy": str.lower}
    _csv_custom_writers = {"toy": str.upper}


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


@pytest.fixture
def csv_data() -> List[CSVPerson]:
    res: List[CSVPerson] = list()
    res.append(
        CSVPerson(
            name="Marie",
            age=0,
            height=1.85,
            woman=True,
            eyes=Eyes.brown,
            hair=Hair.red,
            favorite_func="VLOOKUP()",
        )
    )
    res.append(
        CSVPerson(
            name="Jack Who",
            age=45,
            height=1.43,
            birthday=datetime.fromisoformat("1977-07-23"),
            eyes=Eyes.grey,
            favorite_func="INDEX()",
        )
    )
    res.append(
        CSVPerson(
            name="James 3.5",
            age=18,
            height=1.76,
            birthday=datetime.fromisoformat("2005-02-14"),
            hair=Hair.blonde,
            favorite_func="SUMPRODUCT()",
        )
    )
    return res


@pytest.fixture
def txt_data() -> List[TXTPerson]:
    res: List[TXTPerson] = list()
    res.append(
        TXTPerson(
            name="Marie", age=0, height=1.85, woman=True, eyes=Eyes.brown, hair=Hair.red
        )
    )
    res.append(
        TXTPerson(
            name="Jack Who",
            age=45,
            height=1.43,
            birthday=datetime.fromisoformat("1977-07-23"),
            eyes=Eyes.grey,
        )
    )
    res.append(
        TXTPerson(
            name="James 3.5",
            age=18,
            height=1.76,
            birthday=datetime.fromisoformat("2005-02-14"),
            hair=Hair.blonde,
        )
    )
    return res


@pytest.mark.asyncio
async def test_1_json_exportable_import_save(
    tmp_path: Path, json_parents: List[JSONParent]
):
    fn: Path = tmp_path / "export.json"

    await export(awrap(json_parents), format="json", filename="-")  # type: ignore
    await export(awrap(json_parents), format="json", filename=fn)  # type: ignore
    await export(
        awrap(json_parents), format="json", filename=str(fn.resolve()), force=True
    )  # type: ignore

    imported: set[JSONParent] = set()
    try:
        async for p_in in JSONParent.import_file(fn):
            imported.add(p_in)
    except Exception as err:
        assert False, f"failed to import test data: {err}"

    for data in json_parents:
        try:
            imported.remove(data)
        except Exception as err:
            assert False, f"could not export or import item: {data}: {err}"

    assert len(imported) == 0, "Export or import failed"

    for parent in json_parents:
        assert await parent.save_json(fn) > 0, (
            f"could not save JSONExportable: {str(parent)}"
        )
        assert (parent_imported := await JSONParent.open_json(fn)) is not None, (
            f"could not import save json: {str(parent)}"
        )
        assert parent == parent_imported, (
            f"imported data does not match original: original={parent}, imported={parent_imported}"
        )
        assert (adult := await JSONAdult.open_json(fn)) is None, (
            f"open_json() returned instance even it should not: {adult}"
        )

    parent = json_parents[0]
    assert await parent.save_json(fn.with_suffix("")) > 0, (
        f"could not save JSONExportable: {str(parent)}"
    )

    # Test for opening non-existent file
    assert (
        adult := await JSONAdult.open_json(fn.with_suffix(".not-found.json"))
    ) is None, f"open_json() returned instance from non-existent file: {adult}"


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


def test_6_parse_str_fails(json_parents: List[JSONParent]):
    for parent in json_parents:
        assert (_ := JSONAdult.parse_str(parent.json_src())) is None, (
            f"parse_str() returned instance from faulty data: {parent.json_src()}"
        )


def test_7_jsonexportablerootdict(json_parents: List[JSONParent]):
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


@pytest.mark.asyncio
async def test_8_txt_exportable_importable(tmp_path: Path, txt_data: List[TXTPerson]):
    fn: Path = tmp_path / "export.txt"

    await export(awrap(txt_data), "txt", filename="-")  # type: ignore
    await export(awrap(txt_data), "txt", filename=fn)  # type: ignore
    await export(awrap(txt_data), format="txt", filename=str(fn.resolve()), force=True)  # type: ignore

    imported: set[TXTPerson] = set()
    try:
        async for p_in in TXTPerson.import_file(fn):
            debug("import_txt(): %s", str(p_in))
            imported.add(p_in)
    except Exception as err:
        assert False, f"failed to import test data: {err}"

    assert len(imported) == len(txt_data), (
        f"failed to import all data from TXT file: {len(imported)} != {len(txt_data)}"
    )

    for data in txt_data:
        debug("trying to remove data=%s from imported", str(data))
        try:
            imported.remove(data)
        except Exception as err:
            assert False, f"could not export or import item: {data}: {err}: {imported}"

    assert len(imported) == 0, "Export or import failed"


@pytest.mark.asyncio
async def test_9_csv_exportable_importable(tmp_path: Path, csv_data: List[CSVPerson]):
    fn: Path = tmp_path / "export.csv"

    await export(awrap(csv_data), "csv", filename="-")  # type: ignore
    await export(awrap(csv_data), "csv", filename=fn)  # type: ignore
    await export(awrap(csv_data), "csv", filename=str(fn.resolve()), force=True)  # type: ignore

    imported: set[CSVPerson] = set()
    try:
        async for p_in in CSVPerson.import_file(fn):
            debug("imported: %s", str(p_in))
            imported.add(p_in)
    except Exception as err:
        assert False, f"failed to import test data: {err}"

    assert len(imported) == len(csv_data), (
        f"could not import all CSV data: {len(imported)} != {len(csv_data)}"
    )
    for data_imported in imported:
        debug("hash(data_imported)=%d", hash(data_imported))
        try:
            if data_imported in csv_data:
                ndx: int = csv_data.index(data_imported)
                data = csv_data[ndx]
                assert data == data_imported, (
                    f"imported data different from original: {data_imported} != {data}"
                )
                csv_data.pop(ndx)
            else:
                assert False, f"imported data not in the original: {data_imported}"
        except ValueError:
            assert False, (
                f"export/import conversion error. imported data={data_imported} is not in input data"
            )

    assert len(csv_data) == 0, (
        f"could not import all the data correctly: {len(csv_data)} != 0"
    )


def test_10_PyObjectIdasIdx() -> None:
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
async def test_11_IntasIdx(tmp_path: Path) -> None:
    d = IntIdxExportableDict()
    export_fn: Path = tmp_path / "test_11_IntasIdx.json"
    L: int = 20
    for i in range(L):
        d.add(JSONIntIdxExportable(idx=i))

    debug(d.json_src(exclude_defaults=False))
    assert len(d) == L, f"could not add all the items: {len(d)} != {L}"
    assert await d.save_json(export_fn) > 0, (
        "could not export IntIdxExportableDict to JSON"
    )

    assert (imported := await IntIdxExportableDict.open_json(export_fn)) is not None, (
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
