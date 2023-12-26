import pytest  # type: ignore
from typing import Self, List
from pydantic import Field
from pathlib import Path
from datetime import date, datetime
from enum import StrEnum, IntEnum
import json
import logging
from pydantic_exportables import (
    JSONExportable,
    export,
    Idx,
    CSVExportable,
    TXTExportable,
    TXTImportable,
    Importable,
)
from pyutils import awrap
from pyutils.utils import epoch_now

########################################################
#
# Test Plan
#
########################################################

# 1) Create instances, export as JSON, import back and compare
# 2) Create instances, export as CSV, import back and compare
# 3) Create instances, export as TXT, import back and compare

logger = logging.getLogger()
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
    created: int = Field(default_factory=epoch_now)

    @property
    def index(self) -> Idx:
        """return backend index"""
        return self.name

    @property
    def indexes(self) -> dict[str, Idx]:
        """return backend indexes"""
        return {"name": self.index}


class JSONParent(JSONExportable, Importable):
    name: str
    amount: int = 0
    correct: bool = Field(default=False, alias="c")
    array: List[str] = Field(default_factory=list)
    child: JSONChild | None = Field(default=None)

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
    child: JSONChild | None = Field(default=None)

    def transform2JSONParent(self) -> JSONParent:
        return JSONParent(
            name=self.name, amount=self.age, correct=True, child=self.child
        )


JSONParent.register_transformation(JSONAdult, JSONAdult.transform2JSONParent)


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
def json_data() -> List[JSONParent]:
    c1 = JSONChild(name="c1")
    c3 = JSONChild(name="c3")
    res: List[JSONParent] = list()
    res.append(JSONParent(name="P1", amount=1, array=["one", "two"], child=c1))
    res.append(JSONParent(name="P2", amount=-6, array=["three", "four"]))
    res.append(JSONParent(name="P3", amount=-6, child=c3))
    return res


@pytest.fixture
def json_adults() -> List[JSONAdult]:
    res: List[JSONAdult] = list()
    res.append(JSONAdult(name="Alice", age=35, child=None))
    res.append(JSONAdult(name="Bob", age=38, child=JSONChild(name="Ted")))
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
async def test_1_json_exportable(tmp_path: Path, json_data: List[JSONParent]):
    fn: Path = tmp_path / "export.json"

    await export(awrap(json_data), format="json", filename="-")  # type: ignore
    await export(awrap(json_data), format="json", filename=fn)  # type: ignore
    await export(
        awrap(json_data), format="json", filename=str(fn.resolve()), force=True
    )  # type: ignore

    imported: set[JSONParent] = set()
    try:
        async for p_in in JSONParent.import_file(fn):
            imported.add(p_in)
    except Exception as err:
        assert False, f"failed to import test data: {err}"

    for data in json_data:
        try:
            imported.remove(data)
        except Exception as err:
            assert False, f"could not export or import item: {data}: {err}"

    assert len(imported) == 0, "Export or import failed"


@pytest.mark.asyncio
async def test_2_json_exportable_include_exclude() -> None:
    # test for custom include/exclude
    parent = JSONParent(
        name="P3", amount=-6, correct=False, child=JSONChild(name="test")
    )

    parent_src: dict
    parent_db: dict
    parent_src = json.loads(parent.json_src())
    assert (
        "array" in parent_src
    ), "json_src() failed: _exclude_unset set 'False', 'array' excluded"
    parent_db = json.loads(parent.json_db())
    assert (
        "c" not in parent_db
    ), "json_db() failed: _exclude_defaults set 'True', 'c' included"

    for excl, incl in zip(["child", None], ["name", None]):
        kwargs: dict[str, set[str]] = dict()
        if excl is not None:
            kwargs["exclude"] = {excl}
        if incl is not None:
            kwargs["include"] = {incl}

        parent_src = json.loads(parent.json_src(fields=None, **kwargs))
        parent_db = json.loads(parent.json_db(fields=None, **kwargs))
        if excl is not None:
            assert (
                excl not in parent_db
            ), f"json_src() failed: excluded field {excl} included"
            assert (
                excl not in parent_src
            ), f"json_db() failed: excluded field {excl} included"
        if incl is not None:
            assert (
                incl in parent_src
            ), f"json_src() failed: included field {incl} excluded"
            assert (
                incl in parent_db
            ), f"json_db() failed: included field {incl} excluded"

    parent_src = parent.obj_src(fields=["name", "array"])
    assert (
        "amount" not in parent_src
    ), "json_src() failed: excluded field 'amount' included"
    assert "array" in parent_src, "json_src() failed: included field 'array' excluded"

    parent_db = parent.obj_db(fields=["name", "array"])
    assert (
        "amount" not in parent_db
    ), "json_db() failed: excluded field 'amount' included"
    assert "array" in parent_db, "json_db() failed: included field 'array' excluded"

    parent_src = parent.obj_src()
    assert (
        parent_new := JSONParent.from_obj(parent_src)
    ) is not None, "could not create object from exported model"
    assert (
        parent == parent_new
    ), f"re-created object is different to original: {parent_new}"

    parent_db = parent.obj_db()
    assert (
        parent_new := JSONParent.from_obj(parent_db)
    ) is not None, "could not create object from exported model"
    assert (
        parent == parent_new
    ), f"re-created object is different to original: {parent_new}"


def test_3_jsonexportable_update(json_data: List[JSONParent]):
    """
    test for JSONExportable.update()
    """
    p0: JSONParent = json_data[0]
    p1: JSONParent = json_data[1]
    p2: JSONParent = json_data[2]

    p: JSONParent = p0.model_copy(deep=True)

    for new in json_data[1:]:
        assert not p.update(
            new, match_index=True
        ), "update succeeded even the indexes do not match"
    assert p.update(
        p1, match_index=False
    ), "update did not succeeded even the indexes were ignored"
    assert all(
        [
            p.name == p1.name,
            p.amount == p1.amount,
            p.correct == p1.correct,
            p.array == p1.array,
            p.child == p0.child,
        ]
    ), f"update() failed: updated={str(p)}"
    assert p.update(
        p2, match_index=False
    ), "update did not succeeded even the indexes were ignored"
    assert all(
        [
            p.name == p2.name,
            p.amount == p2.amount,
            p.correct == p2.correct,
            p.array == p1.array,
            p.child == p2.child,
        ]
    ), f"update() failed: updated={str(p)}"


def test_4_jsonexportable_transform(json_adults: List[JSONAdult]):
    res = JSONParent.transform_many(json_adults)
    assert len(res) == len(
        json_adults
    ), f"could not transform all data: {len(res)} != {len(json_adults)}"


@pytest.mark.asyncio
async def test_5_txt_exportable_importable(tmp_path: Path, txt_data: List[TXTPerson]):
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

    assert len(imported) == len(
        txt_data
    ), f"failed to import all data from TXT file: {len(imported)} != {len(txt_data)}"

    for data in txt_data:
        debug("trying to remove data=%s from imported", str(data))
        try:
            imported.remove(data)
        except Exception as err:
            assert False, f"could not export or import item: {data}: {err}: {imported}"

    assert len(imported) == 0, "Export or import failed"


@pytest.mark.asyncio
async def test_6_csv_exportable_importable(tmp_path: Path, csv_data: List[CSVPerson]):
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

    assert len(imported) == len(
        csv_data
    ), f"could not import all CSV data: {len(imported)} != {len(csv_data)}"
    for data_imported in imported:
        debug("hash(data_imported)=%d", hash(data_imported))
        try:
            if data_imported in csv_data:
                ndx: int = csv_data.index(data_imported)
                data = csv_data[ndx]
                assert (
                    data == data_imported
                ), f"imported data different from original: {data_imported} != {data}"
                csv_data.pop(ndx)
            else:
                assert False, f"imported data not in the original: {data_imported}"
        except ValueError:
            assert (
                False
            ), f"export/import conversion error. imported data={data_imported} is not in input data"

    assert (
        len(csv_data) == 0
    ), f"could not import all the data correctly: {len(csv_data)} != 0"
