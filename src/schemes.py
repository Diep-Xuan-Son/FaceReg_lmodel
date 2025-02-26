from typing import List, Union
from pydantic import BaseModel, model_validator
from fastapi import Query, File, UploadFile
import json

class Personfile(BaseModel):
	images:	List[UploadFile] = File(...)

class Person(BaseModel):
	code: str
	name: str
	birthday: str

	# @model_validator(mode='before')
	# @classmethod
	# def validate_to_json(cls, value):
	# 	if isinstance(value, str):
	# 		return cls(**json.loads(value))
	# 	return value

	@classmethod
	def as_form(
		cls,
		code: str = Query("001099008839", description="ID of person, ID is unique"),
		name: str = Query("Son", description="Name of person"),
		birthday: str = Query("29/04/1999", description="Birthday of person")
	):
		return cls(code=code, name=name, birthday=birthday)
