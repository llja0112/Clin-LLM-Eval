"""Generation models for formatting of LLM outputs"""

from typing import List
from pydantic import BaseModel, Field

class CaseDetailsFormat(BaseModel):
  "Evaluation metric for differential diagnosis field"

  basic_details: str = Field(
    description="Case basic details")

  vitals: str = Field(
    description="Vitals of patient"
  )

  physical_presentation: str = Field(
    description="Physical presentation of patient"
  )

  challenging_question: str = Field(
    description = "Challenging question to ask"
  )

class HistoryTakingChecklistItem(BaseModel):
  "History taking checklist format"

  question: str = Field (description='Category of question')
  response: str = Field (description='Patient response')

class HistoryTakingChecklist(BaseModel):
  "Evaluation metric for differential diagnosis field"

  ChecklistItems: List[HistoryTakingChecklistItem] = Field(
    description="History taking checklist items")

class PhysicalExamChecklistItem(BaseModel):
  "Physical exam checklist format"

  technique: str = Field (description='Medical physical exam techniques')
  justification: str = Field (
    description='Justification of why this physical exam technique should be conducted')

class PhysicalExamChecklist(BaseModel):
  "Checklist of physical exam techniques to complete in a medical consult"

  ChecklistItems: List[PhysicalExamChecklistItem] = Field(
    description="Physical exam checklist items")

class InvestigationsChecklistItem(BaseModel):
  "Investigations checklist format"

  investigation: str = Field (description='Investigation to identify diagnosis')
  justification: str = Field (
    description='Justification for why a particular diagnosis is selected')

class InvestigationsChecklist(BaseModel):
  "Checklist for investigations that doctor should suggest"

  ChecklistItems: List[InvestigationsChecklistItem] = Field(
    description="Investigations checklist items")

class DdxChecklistItem(BaseModel):
  "Ddx checklist format"

  diagnosis: str = Field (description='Possible differential diagnosis')
  justification: str = Field (
    description='Justification of why this differential diagnosis was suggested')

class DdxChecklist(BaseModel):
  "Checklist of differential diagnoses which physician should suggest"

  ChecklistItems: List[DdxChecklistItem] = Field(
    description="Differential diagnoses checklist items")
