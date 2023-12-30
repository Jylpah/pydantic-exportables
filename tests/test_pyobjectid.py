import pytest  # type: ignore

from bson import ObjectId
from typing import List

from pydantic_exportables import (
    PyObjectId,
    validate_object_id,
)


@pytest.fixture
def object_ids() -> List[PyObjectId]:
    return [ObjectId() for _ in range(10)]


@pytest.fixture
def object_id_strs_ok() -> List[str]:
    return [str(ObjectId()) for _ in range(10)]


@pytest.fixture
def object_id_strs_nok() -> List[str]:
    return (
        [str(ObjectId())[3:-1] for _ in range(4)]
        + [str(ObjectId()) + "SHOULD_FAIL" for _ in range(4)]
        + ["Thzsz orz not HEX -1"]
    )


def test_1_PyObjectId(object_ids: List[PyObjectId]):
    for object_id in object_ids:
        assert isinstance(
            validate_object_id(object_id), ObjectId
        ), f"could not validate a valid ObjectId: {str(object_id)}"


def test_2_validate_object_id_str_OK(object_id_strs_ok: List[str]):
    for object_id_str in object_id_strs_ok:
        assert isinstance(
            validate_object_id(object_id_str), ObjectId
        ), f"could not validate a valid ObjectId string: {object_id_str}"


def test_3_validate_object_id_str_FAIL(object_id_strs_nok: List[str]):
    for object_id_str in object_id_strs_nok:
        try:
            validate_object_id(object_id_str)
            assert False, f"validated an invalida ObjectId string: {object_id_str}"
        except ValueError:
            pass
