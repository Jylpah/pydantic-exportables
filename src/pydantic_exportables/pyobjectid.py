## Copyright PhilJay, 2023
# source https://stackoverflow.com/a/77105412

from typing import Any
from bson import ObjectId
from pydantic_core import core_schema, CoreSchema
from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue

# class PyObjectId(str):
#     @classmethod
#     def __get_pydantic_core_schema__(
#         cls, _source_type: Any, _handler: Any
#     ) -> core_schema.CoreSchema:
#         return core_schema.json_or_python_schema(
#             json_schema=core_schema.str_schema(),
#             python_schema=core_schema.union_schema(
#                 [
#                     core_schema.is_instance_schema(ObjectId),
#                     core_schema.chain_schema(
#                         [
#                             core_schema.str_schema(),
#                             core_schema.no_info_plain_validator_function(cls.validate),
#                         ]
#                     ),
#                 ]
#             ),
#             serialization=core_schema.plain_serializer_function_ser_schema(
#                 lambda x: str(x)
#             ),
#         )

#     @classmethod
#     def validate(cls, value) -> ObjectId:
#         if not ObjectId.is_valid(value):
#             raise ValueError("Invalid ObjectId")

#         return ObjectId(value)


class PyObjectId(ObjectId):
    """Wrapper for ObjectId"""

    # https://stackoverflow.com/a/77101754/12946084
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def validate(value: str) -> ObjectId:
            if not ObjectId.is_valid(value):
                raise ValueError("Invalid ObjectId")
            return ObjectId(value)

        return core_schema.no_info_plain_validator_function(
            function=validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())
