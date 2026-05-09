from pathlib import Path
from typing import List

import pytest

from enum import Enum
from datetime import date, datetime
from pydantic_exportables import CSVExportable, export_csv


class Gender(Enum):
    male = 1
    female = 2


class CSVPerson(CSVExportable):
    name: str
    bday: date | None = None
    gender: Gender | None = None
    height: float | None = None
    weight: int | None = None


class CSVParent(CSVPerson):
    spouse: CSVPerson | None = None
    married: datetime | None = None
    child: CSVPerson | None = None

    def set_marriage(self, spouse: CSVPerson, wedding: datetime) -> None:
        self.spouse = spouse
        self.married = wedding


@pytest.fixture
def csv_persons() -> List[CSVPerson]:
    res: list[CSVPerson] = list()
    child: CSVPerson = CSVPerson(
        name="Lisa",
        bday=date(2010, 3, 20),
        gender=Gender.female,
        height=1.51,
        weight=45,
    )
    matt = CSVParent(
        name="Matt", bday=date(1982, 8, 25), gender=Gender.male, height=1.75, weight=70
    )
    jane = CSVParent(
        name="Jane",
        bday=date(1983, 5, 30),
        gender=Gender.female,
        height=1.68,
        weight=65,
    )
    wedding_date: datetime = datetime(2005, 6, 15)
    matt.set_marriage(jane, wedding=wedding_date)
    jane.set_marriage(matt, wedding=wedding_date)
    matt.child = child
    jane.child = child

    res.append(matt)
    res.append(jane)
    res.append(
        CSVPerson(
            name="Ada",
            bday=date(1985, 12, 10),
            gender=Gender.female,
            height=1.65,
            weight=60,
        )
    )
    res.append(
        CSVPerson(
            name="Bob",
            bday=date(1980, 6, 15),
            gender=Gender.male,
            height=1.80,
            weight=80,
        )
    )

    return res


@pytest.mark.asyncio
async def test_csvexportable_export_import(
    tmp_path: Path, csv_persons: List[CSVPerson]
) -> None:
    filename = tmp_path / "people.csv"

    await export_csv(filename=filename, iterable=csv_persons, force=True)

    imported = [item async for item in CSVParent.import_csv(filename)]

    assert len(imported) == len(csv_persons), (
        "different number of items imported than original"
    )
    for i in range(len(csv_persons)):
        assert imported[i] == CSVParent.model_validate(csv_persons[i].model_dump()), (
            f"exported item {i} was not successfully imported: {str(csv_persons[i])}"
        )


@pytest.mark.asyncio
async def test_export_csv_writes_to_stdout_for_dash_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    people = [CSVPerson(name="Ada"), CSVPerson(name="Bob")]

    exported, errors = await export_csv(filename="-", iterable=people)

    assert (exported, errors) == (2, 0)
    assert capsys.readouterr().out == (
        "name,bday,gender,height,weight\n"
        "Ada,,,,\n"
        "Bob,,,,\n"
    )
    assert not (tmp_path / "-").exists()


@pytest.mark.asyncio
async def test_utils_import_csv_wrapper(tmp_path: Path) -> None:
    filename = tmp_path / "people.csv"
    filename.write_text("name,age,child.name\nAda,37,Eve\n", encoding="utf-8")

    imported = [item async for item in CSVParent.import_csv(filename)]

    assert imported == [CSVParent(name="Ada", age=37, child=CSVPerson(name="Eve"))]
