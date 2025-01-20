"""Case generator for clinical reasoning evaluation"""

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

import pandas as pd

from . import generation_models

class CaseGenerator:
  """Generate clinical cases for dynamic testing of LLM models"""

  def __init__(
      self, model_name="llama3-8b",
      case_stem="",
      basic_details_example="",
      vitals_example="",
      physical_presentation_example="",
      challenging_question_example="",
      history_taking_example=""):
    self.model_name = model_name
    self.temperature = 0
    self.initialize_environment()
    self.model = None
    self.chat_model = None
    self.select_model()

    self.case_stem = case_stem
    self.basic_details_example = basic_details_example
    self.vitals_example = vitals_example
    self.physical_presentation_example = physical_presentation_example
    self.challenging_question_example = challenging_question_example
    self.history_taking_example = history_taking_example

    self.basic_details = None
    self.vitals = None
    self.physical_presentation = None
    self.challenging_question = None

    self.history = None
    self.history_string = None
    self.physical_exam = None
    self.investigations = None
    self.ddx = None

  def initialize_environment(self):
    """Load environment variables for LLM API calls"""
    load_dotenv()

  def select_model(self):
    """Method selecting model for patient agent"""
    if self.model_name.lower() == 'llama3-8b':
      self.chat_model = ChatOllama(model="llama3.1:8b", temperature=self.temperature)
      self.model = OllamaLLM(model="llama3.1:8b", temperature=self.temperature)
    elif self.model_name.lower() == 'gpt-4o':
      self.chat_model = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
      self.model = OpenAI(model="gpt-4o", temperature=self.temperature)
    elif self.model_name.lower() == 'gpt-4o-mini':
      self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=self.temperature)
      self.model = OpenAI(model="gpt-4o-mini", temperature=self.temperature)
    else:
      raise ValueError(f"Unknown model name: {self.model_name}")

  def generate_basic_details(self):
    """Generate basic details of clinical case"""
    template = """
    You are an expert clinician who is creating patient case scenarios to test a new doctor
    on his clinical skills e.g. history taking, physical exam, investigations and differential diagnosis.

    Generate a case basic details, vitals, physical presentation and challenging question for the following scenario "{case_stem}" based on the example provided below.

    Example case basic details:
    {case_component_example}

    Example vitals:
    {vitals_example}

    Example physical presentation:
    {physical_presentation_example}

    Example challenging question:
    {challenging_question_example}
    """

    prompt = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(generation_models.CaseDetailsFormat)

    chain = prompt | structured_model

    output = chain.invoke({
      "case_stem": self.case_stem,
      "case_component_example": self.basic_details_example,
      "vitals_example": self.vitals_example,
      "physical_presentation_example": self.physical_presentation_example,
      "challenging_question_example": self.challenging_question_example,
    })

    self.basic_details = output.basic_details
    self.vitals = output.vitals
    self.physical_presentation = output.physical_presentation
    self.challenging_question = output.challenging_question

    return [self.basic_details, self.vitals, self.physical_exam, self.challenging_question]

  def generate_history(self):
    """Generate history based on basic details of clinical case"""

    template = """
    You are an expert clinician who is creating patient case scenarios to test a new doctor
    on his clinical skills e.g. history taking, physical exam, investigations and differential diagnosis.

    You are provided with the following basic details, vitals, physical presentation.

    Basic details:
    {basic_details}

    Vitals:
    {vitals}

    Physical Presentation:
    {physical_presentation}

    Generate a history taking checklist based on the example provided below.

    History taking checklist example:
    {history_taking_example}
    """

    prompt = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(
      generation_models.HistoryTakingChecklist
    )

    chain = prompt | structured_model

    self.history = chain.invoke({
      "basic_details": self.basic_details,
      "vitals": self.vitals,
      "physical_presentation": self.physical_presentation,
      "history_taking_example": self.history_taking_example
    })

    self.history_string = "\n".join(
      [f"Question: {item.question} | Response: {item.response}"
      for item in self.history.ChecklistItems])

    return self.history

  def generate_physical_exam(self):
    """Generate physical exam based on details of clinical case"""

    template = """
    You are an expert clinician who is creating patient case scenarios to test a new doctor
    on his clinical skills e.g. history taking, physical exam, investigations and differential diagnosis.

    You are provided with the following basic details, vitals, physical presentation and history.

    Basic details:
    {basic_details}

    Vitals:
    {vitals}

    Physical Presentation:
    {physical_presentation}

    History:
    {history}

    Generate a checklist of physical exam techniques a doctor should perfrom.
    """

    prompt = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(
      generation_models.PhysicalExamChecklist
    )

    chain = prompt | structured_model

    self.physical_exam = chain.invoke({
      "basic_details": self.basic_details,
      "vitals": self.vitals,
      "physical_presentation": self.physical_presentation,
      "history": self.history_string
    })

    return self.physical_exam

  def generate_investigations(self):
    """Generate investigations based on details of clinical case"""
    template = """
    You are an expert clinician who is creating patient case scenarios to test a new doctor
    on his clinical skills e.g. history taking, physical exam, investigations and differential diagnosis.

    You are provided with the following basic details, vitals, physical presentation and history.

    Basic details:
    {basic_details}

    Vitals:
    {vitals}

    Physical Presentation:
    {physical_presentation}

    History:
    {history}

    Generate a checklist of investigations a doctor should perfrom.
    """

    prompt = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(
      generation_models.InvestigationsChecklist
    )

    chain = prompt | structured_model

    self.investigations = chain.invoke({
      "basic_details": self.basic_details,
      "vitals": self.vitals,
      "physical_presentation": self.physical_presentation,
      "history": self.history_string
    })

    return self.investigations

  def generate_ddx(self):
    """Generate Ddx based on details of clinical case"""
    template = """
    You are an expert clinician who is creating patient case scenarios to test a new doctor
    on his clinical skills e.g. history taking, physical exam, investigations and differential diagnosis.

    You are provided with the following basic details, vitals, physical presentation and history.

    Basic details:
    {basic_details}

    Vitals:
    {vitals}

    Physical Presentation:
    {physical_presentation}

    History:
    {history}

    Generate a checklist of possible differentail diagnoses a doctor should suggest.
    """

    prompt = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(
      generation_models.DdxChecklist
    )

    chain = prompt | structured_model

    self.ddx = chain.invoke({
      "basic_details": self.basic_details,
      "vitals": self.vitals,
      "physical_presentation": self.physical_presentation,
      "history": self.history_string
    })

    return self.ddx

  def export_case(self, verbose=True, dir_name="Data/", file_name="1"):
    "Export generated case"

    print(">>> Exporting basic details")
    details_df = pd.DataFrame()
    details_df.at[0, 'Basic details'] = self.basic_details
    details_df.at[0, 'Vitals'] = self.vitals
    details_df.at[0, 'Physical presentation'] = self.physical_presentation
    details_df.at[0, 'Challenging question'] = self.challenging_question
    details_df.to_csv(dir_name + "details/" + file_name + ".csv")

    if verbose:
      print(">>> Exporting history")
    columns = ["Question", "Patient Response"]
    history_df = pd.DataFrame(columns=columns)

    for item in self.history.ChecklistItems:
      history_df = pd.concat(
        [
          history_df,
          pd.DataFrame(
            [[
              item.question, item.response
            ]], columns=columns)
        ]
      )

    history_df.reset_index(drop=True)
    history_df.to_csv(dir_name + "history/" + file_name + ".csv")

    if verbose:
      print(">>> Exporting Physical Exam")
    columns = ["Physical Exam", "Justification"]
    physical_exam_df = pd.DataFrame(columns=columns)

    for item in self.physical_exam.ChecklistItems:
      physical_exam_df = pd.concat(
        [
          physical_exam_df,
          pd.DataFrame(
            [[
              item.technique, item.justification
            ]], columns=columns)
        ]
      )

    physical_exam_df.reset_index(drop=True)
    physical_exam_df.to_csv(dir_name + "physical/" + file_name + ".csv")

    if verbose:
      print(">>> Exporting Investigations")
    columns = ["Investigations", "Justification"]
    investigations_df = pd.DataFrame(columns=columns)

    for item in self.investigations.ChecklistItems:
      investigations_df = pd.concat(
        [
          investigations_df,
          pd.DataFrame(
            [[item.investigation, item.justification
              ]], columns=columns)
        ]
      )

    investigations_df.reset_index(drop=True)
    investigations_df.to_csv(dir_name + "investigations/" + file_name + ".csv")

    if verbose:
      print(">>> Exporting Differential Diagnosis")
    columns = ["Diagnosis", "Justification"]
    diagnoses_df = pd.DataFrame(columns=columns)

    for item in self.ddx.ChecklistItems:
      diagnoses_df = pd.concat([
        diagnoses_df,
        pd.DataFrame(
          [[
            item.diagnosis, item.justification
          ]], columns=columns)
      ])

    diagnoses_df.reset_index(drop=True)
    diagnoses_df.to_csv(dir_name + "diagnosis/" + file_name + ".csv")
