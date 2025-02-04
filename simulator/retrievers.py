"""Prompt templates retriever"""

from langchain_core.prompts import PromptTemplate

HISTORY_EVAL_TEMPLATE = """
You are an expert evaluator of doctors taking history from a patient.

This is the conversation between the doctor and the patient:
{conversation}

Now make an assessment if the doctor asked questions pertaining to '{history_taking_category}' to get the following '{patient_response}'
"""

PHYSICAL_EVAL_TEMPLATE = """
You are an expert evaluator of doctors performing physical examinations on a patient.

These are the physical examination steps suggested by the doctor:
{physical_exam_suggestions}

Now make an assessment if the doctor suggested any physical examp step similar to '{physical_exam_step}' 
with a similar justification '{physical_exam_justification}'
"""

INVESTIGATIONS_EVAL_TEMPLATE = """
You are an expert evaluator of doctors suggesting investigations for a patient.

These are the investigations suggested by the doctor:
{investigations_suggestions}

Now make an assessment if the doctor suggested any investigations similar to '{investigations_item}' 
with a similar justification '{investigations_justification}'
"""

DIAGNOSIS_EVAL_TEMPLATE = """
You are an expert evaluator of doctors providing differential diagnoses for a patient.

These are the differential diagnoses suggested by the doctor:
{ddx_suggestions}

Now make an assessment if the doctor suggested any diagnosis similar to '{ddx_item}'
"""

class TemplateStore:
  """Template Store"""

  def __init__(self):
    pass

  @staticmethod
  def get_prompt_template(prompt_type):
    """Get prompt template"""
    if prompt_type == "history":
      return PromptTemplate.from_template(HISTORY_EVAL_TEMPLATE)
    elif prompt_type == "physical":
      return PromptTemplate.from_template(PHYSICAL_EVAL_TEMPLATE)
    elif prompt_type == "investigations":
      return PromptTemplate.from_template(INVESTIGATIONS_EVAL_TEMPLATE)
    elif prompt_type == "diagnosis":
      return PromptTemplate.from_template(DIAGNOSIS_EVAL_TEMPLATE)
    else:
      return None
