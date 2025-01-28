"""Index module containing multiple agent modules"""

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_community.llms import Replicate

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd

from . import generation_models

class PatientAgent:
  """Class representing a patient agent"""

  def __init__(self, model_name="llama3-8b", thread_id="1", details=None, history=None):
    self.model_name = model_name
    self.temperature = 0
    self.initialize_environment()
    self.config = {"configurable": {"thread_id": thread_id}}
    self.chat_model = None
    self.select_model()
    self.agent = None
    self.step = 0
    self.template = '''
    You are taking on a role of a patient as part of a doctor simulation training.
    The patient has the following case description: 
    {case_prompt}

    The patient has the following vitals:
    {vitals}

    You are going to answer questions from a doctor about your symptoms and here are 
    some possible answers you can give depending on the category of the questions:
    {question_answer_list}

    When asked if there are any questions, ask the following challenging question:
    {challenging_question}

    If you do not know the answer to the question, you can say "I don't know".
    If you are asked questions unrelated to your role as the patient, you will say "I do not understand why you ask this question."
    Always answer the question in human dialogue format with a maximum of 3 sentences.
    '''

    if details is not None and history is not None:
      question_answer_list = "\n".join(
        [f"Question: {row['Question']} | Answer: {row['Patient Response']}"
         for index, row in history.iterrows()])

      self.system_prompt = self.template.format(
        case_prompt=details['Basic details'].values[0],
        vitals=details['Vitals'].values[0],
        question_answer_list = question_answer_list,
        challenging_question = details['Challenging question'].values[0]
      )
    else:
      self.system_prompt = '''
      You are taking on a role of a patient as part of a doctor simulation training.
      '''

  def initialize_environment(self):
    """Initialize and get secret keys for LLM models"""
    load_dotenv()

  def select_model(self):
    """Method selecting model for patient agent"""
    if self.model_name.lower() == 'llama3-8b':
      self.chat_model = Replicate(model="meta/meta-llama-3-8b",
                model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'llama3-70b':
      self.chat_model = Replicate(model="meta/meta-llama-3-70b",
                model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'gpt-4o':
      self.chat_model = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
    elif self.model_name.lower() == 'gpt-4o-mini':
      self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-flash':
      self.chat_model = ChatVertexAI(model="gemini-1.5-flash", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-pro':
      self.chat_model = ChatVertexAI(model="gemini-1.5-pro", temperature=self.temperature)
    else:
      raise ValueError(f"Unknown model name: {self.model_name}")

  def compile(self):
    """Simulate a patient agent"""

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the function that calls the model
    def call_model(state: MessagesState):
      messages = state['messages']
      response = self.chat_model.invoke(messages)
      return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    self.agent = workflow.compile(checkpointer=memory)

  def invoke(self, message):
    """Invoke LLM model"""
    if self.step == 0:
      input_messages = [SystemMessage(self.system_prompt), HumanMessage(message)]
    else:
      input_messages = [HumanMessage(message)]

    output = self.agent.invoke({"messages": input_messages}, config=self.config)
    self.step += 1
    return output

class DoctorAgent:
  """Class representing a doctor agent"""

  def __init__(self, model_name="llama3-8b", thread_id="1", intro_message="Hi! I am Dr Medbot."):
    self.model_name = model_name
    self.temperature = 0
    self.initialize_environment()
    self.config = {"configurable": {"thread_id": thread_id}}
    self.model = None
    self.chat_model = None
    self.select_model()
    self.agent = None
    self.step = 0
    self.intro_message = intro_message
    self.system_prompt = '''
    You are taking on a role of a doctor taking a complete medical history from a patient.
    Ask the patient one question at at time.
    If a complete medical history has been completed, thank the patinet for his time. 
    '''
    self.history = None
    self.physical_exam = None
    self.investigations = None
    self.differential_diagnoses = None

  def initialize_environment(self):
    """Initialize and get secret keys for LLM models"""
    load_dotenv()

  def select_model(self):
    """Method selecting model for doctor agent"""
    if self.model_name.lower() == 'llama3-8b':
      self.chat_model = Replicate(model="meta/meta-llama-3-8b",
                                  model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'llama3-70b':
      self.chat_model = Replicate(model="meta/meta-llama-3-70b",
                                  model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'gpt-4o':
      self.chat_model = ChatOpenAI(model="gpt-4o", temperature=self.temperature)
    elif self.model_name.lower() == 'gpt-4o-mini':
      self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-flash':
      self.chat_model = ChatVertexAI(model="gemini-1.5-flash", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-pro':
      self.chat_model = ChatVertexAI(model="gemini-1.5-pro", temperature=self.temperature)
    else:
      raise ValueError(f"Unknown model name: {self.model_name}")

  def compile(self):
    """Simulate a doctor agent"""

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the function that calls the model
    def call_model(state: MessagesState):
      messages = state['messages']
      response = self.chat_model.invoke(messages)
      return {"messages": response}

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    self.agent = workflow.compile(checkpointer=memory)

  def invoke(self, message):
    """Invoke LLM model"""

    if self.step == 0:
      input_messages = [SystemMessage(self.system_prompt),
                        AIMessage(self.intro_message),
                        HumanMessage(message)]
    else:
      input_messages = [HumanMessage(message)]

    self.history = self.agent.invoke({"messages": input_messages}, config=self.config)
    self.step += 1
    return self.history

  def combine_history_string(self):
    """Combine history taken into a single string"""

    messages = self.agent.get_state(self.config).values['messages']
    filtered_messages = [message for message in messages if not isinstance(message, SystemMessage)]
    labeled_messages = [
      "Doctor: " + message.content if isinstance(message, AIMessage) else 
      "Human: " + message.content for message in filtered_messages]
    return "\n".join(labeled_messages)

  def invoke_physical_exam(self):
    """Invoke physical exam steps that doctor should take"""

    template = """
    You are taking on a role of a doctor. 
    Your conversation with the patient for history taking is as follows:
    {patient_history}

    Suggest the physical examination you would like to perform as a doctor.
    The suggestion should not include any investigations like ECG, CXR or blood tests.
    """

    combined_messages = self.combine_history_string()


    if "llama" in self.model_name:
      # parser = PydanticOutputParser(generation_models.PhysicalExamChecklist)

      return "Not implemented"

    structured_model = self.chat_model.with_structured_output(
      generation_models.PhysicalExamChecklist)

    prompt_template = PromptTemplate.from_template(template)
    chain = prompt_template | structured_model
    output = self.physical_exam = chain.invoke({"patient_history": combined_messages})
    return output

  def invoke_investigations(self):
    """Invoke the investigations that doctor should suggest"""

    template = """
    You are taking on a role of a doctor. 
    Your conversation with the patient for history taking is as follows:
    {patient_history}

    Suggest the investigations you would like to perform as a doctor.
    """

    combined_messages = self.combine_history_string()

    prompt_template = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(
      generation_models.InvestigationsChecklist)

    chain = prompt_template | structured_model
    self.investigations = chain.invoke({"patient_history": combined_messages})

  def invoke_differential_diagnoses(self):
    """Invoke the investigations that doctor should suggest"""

    template = """
    You are taking on a role of a doctor. 
    Your conversation with the patient for history taking is as follows:
    {patient_history}

    Suggest the top 3 differential diagnoses.
    """

    combined_messages = self.combine_history_string()

    prompt_template = PromptTemplate.from_template(template)

    structured_model = self.chat_model.with_structured_output(generation_models.DdxChecklist)

    chain = prompt_template | structured_model
    self.differential_diagnoses = chain.invoke({"patient_history": combined_messages})

  def export_history(self, verbose=True, dir_name="Data/output", file_name="1"):
    """Export history taken by doctor"""

    if verbose:
      print("Exporting history taken by doctor")

    messages_df = pd.DataFrame([
      {"Type": type(message).__name__, "Message": message.content}
      for message in self.agent.get_state(self.config).values['messages']
    ])

    messages_df.to_csv(dir_name + "/history/" + file_name + ".csv", index=False)

  def export_physical_exam(self, verbose=True, dir_name="Data/output", file_name="1"):
    """Export physical exam steps suggested by doctor"""

    if verbose:
      print("Exporting physical exam steps suggested by doctor")

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

    physical_exam_df = physical_exam_df.reset_index(drop=True)
    physical_exam_df.to_csv(dir_name + "/physical/" + file_name + ".csv")

  def export_investigations(self, verbose=True,
                            dir_name="Data/output", file_name="1"):
    """Export investigations suggested by doctor"""

    if verbose:
      print("Exporting investigations suggested by doctor")

    columns = ["Investigation", "Justification"]
    investigations_df = pd.DataFrame(columns=columns)

    for item in self.investigations.ChecklistItems:
      investigations_df = pd.concat(
        [
          investigations_df,
          pd.DataFrame(
            [[
              item.investigation, item.justification
            ]], columns=columns)
        ]
      )

    investigations_df = investigations_df.reset_index(drop=True)
    investigations_df.to_csv(dir_name + "/investigations/" + file_name + ".csv")

  def export_differential_diagnoses(self, verbose=True,
                                    dir_name="Data/output", file_name="1"):
    """Export differential diagnoses suggested by doctor"""

    if verbose:
      print("Exporting differential diagnoses suggested by doctor")

    columns = ["Diagnosis", "Justification"]
    ddx_df = pd.DataFrame(columns=columns)

    for item in self.differential_diagnoses.ChecklistItems:
      ddx_df = pd.concat(
        [
          ddx_df,
          pd.DataFrame(
            [[
              item.diagnosis, item.justification
            ]], columns=columns)
        ]
      )

    ddx_df = ddx_df.reset_index(drop=True)
    ddx_df.to_csv(dir_name + "/diagnosis/" + file_name + ".csv")

  def export_all(self, verbose=True, dir_name="Data/output", file_name="1"):
    """Export all history, physical exam, investigations and differential diagnoses"""

    self.export_history(verbose, dir_name, file_name)
    self.export_physical_exam(verbose, dir_name, file_name)
    self.export_investigations(verbose, dir_name, file_name)
    self.export_differential_diagnoses(verbose, dir_name, file_name)

class HistoryEvaluation(BaseModel):
  "Evaluation metric for history taking field"

  score: int = Field(
    description="If the relevant information is obtained by the doctor \
      in the history taking conversation, give a score of 1. Else, 0.")
  explanation: str = Field(
    description="Explain in 1 sentence why the score was given.")

class PhysicalExamEvaluation(BaseModel):
  "Evaluation metric for physical exam field"
  score: int = Field(
    description="If the relevant physical examination step is suggested \
      by the doctor, give a score of 1. Else, 0.")
  explanation: str = Field(
    description="Explain in 1 sentence why the score was given.")

class InvestigationsEvaluation(BaseModel):
  "Evaluation metric for investigations field"
  score: int = Field(
    description="If the relevant investigations step is suggested \
      by the doctor, give a score of 1. Else, 0.")
  explanation: str = Field(
    description="Explain in 1 sentence why the score was given.")

class DdxEvaluation(BaseModel):
  "Evaluation metric for differential diagnosis field"
  score: int = Field(
    description="If the relevant differential diagnosis is suggested \
      by the doctor, give a score of 1. Else, 0.")
  explanation: str = Field(
    description="Explain in 1 sentence why the score was given.")

class ExaminerAgent:
  """Class representing an examiner agent"""

  def __init__(self, model_name="llama3-8b"):
    self.model_name = model_name
    self.temperature = 0
    self.history_score = None
    self.history_scoring_log = []
    self.investigations_score = None
    self.investigations_scoring_log = []
    self.physical_exam_score = None
    self.physical_exam_scoring_log = []
    self.differential_diagnoses_score = None
    self.differential_diagnoses_scoring_log = []
    self.model = None
    self.select_model()

  def initialize_environment(self):
    """Initialize and get secret keys for LLM models"""
    load_dotenv()

  def select_model(self):
    """Select model for examiner agent"""

    if self.model_name.lower() == 'llama3-8b':
      self.model = Replicate(model="meta/meta-llama-3-8b",
                             model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'llama3-70b':
      self.model = Replicate(model="meta/meta-llama-3-70b",
                             model_kwargs={"temperature": self.temperature})
    elif self.model_name.lower() == 'gpt-4o-mini':
      self.model = ChatOpenAI(model="gpt-4o-mini", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-flash':
      self.chat_model = ChatVertexAI(model="gemini-1.5-flash", temperature=self.temperature)
    elif self.model_name.lower() == 'gemini-pro':
      self.chat_model = ChatVertexAI(model="gemini-1.5-pro", temperature=self.temperature)
    else:
      raise ValueError(f"Unknown model name: {self.model_name}")

  def evaluate_history_taking(self, conversation, history_taking_checklist):
    """Evaluate history taking capability of doctor based on checklist"""

    total_score = 0
    history_scoring_log = []
    for _, history_taking_item in history_taking_checklist.iterrows():
      output = self.evaluate_history_taking_item(
        conversation, history_taking_item['Question'],
        history_taking_item['Patient Response']
      )
      total_score += output.score
      history_scoring_log.append(output.explanation)

    self.history_score = total_score
    self.history_scoring_log = history_scoring_log

  def evaluate_history_taking_item(self, conversation, history_taking_category, patient_response):
    """Evaluate conversation for one history taking item on checklist"""

    template = """
    You are an expert evaluator of doctors taking history from a patient.

    This is the conversation between the doctor and the patient:
    {conversation}

    Now make an assessment if the doctor asked questions pertaining to '{history_taking_category}' to get the following '{patient_response}'
    """

    prompt = PromptTemplate.from_template(template)

    structured_llm = self.model.with_structured_output(HistoryEvaluation)

    chain = prompt | structured_llm

    output = chain.invoke({
      "conversation": conversation, 
      "history_taking_category": history_taking_category, 
      "patient_response": patient_response
    })

    return output

  def evaluate_physical_exam_checklist(self, physical_exam_suggestions, physical_exam_checklist):
    """Evaluate physical exam suggestions based on physical exam checklist"""

    total_score = 0
    physical_exam_scoring_log = []
    for _, physical_exam_item in physical_exam_checklist.iterrows():
      output = self.evaluate_physical_exam_item(
        physical_exam_suggestions,
        physical_exam_item['Exam component'],
        physical_exam_item['Maneuver']
      )
      total_score += output.score
      physical_exam_scoring_log.append(output.explanation)

    self.physical_exam_score = total_score
    self.physical_exam_scoring_log = physical_exam_scoring_log

  def evaluate_physical_exam_item(self, physical_exam_suggestions,
                                  physical_exam_category, physical_exam_details):
    """Evaluate physical exam suggestion for one physical exam item on checklist"""

    template = """
    You are an expert evaluator of doctors performing physical examinations on a patient.

    These are the physical examination steps suggested by the doctor:
    {physical_exam_suggestions}

    Now make an assessment if the doctor suggested the appropriate steps for '{physical_exam_category}' to get the following details '{physical_exam_details}'
    """

    prompt = PromptTemplate.from_template(template)

    structured_llm = self.model.with_structured_output(PhysicalExamEvaluation)

    chain = prompt | structured_llm

    output = chain.invoke({
      "physical_exam_suggestions": physical_exam_suggestions,
      "physical_exam_category": physical_exam_category,
      "physical_exam_details": physical_exam_details
    })

    return output

  def evaluate_investigations_checklist(self, investigations_suggestions, investigations_checklist):
    """Evaluate investigations suggestions with investigations checklist"""

    investigations_score = 0
    investigations_scoring_log = []
    for _, investigations_item in investigations_checklist.iterrows():
      output = self.evaluate_investigations_item(
        investigations_suggestions,
        investigations_item['Diagnostic Workup'])
      investigations_score += output.score
      investigations_scoring_log.append(output.explanation)

    self.investigations_score = investigations_score
    self.investigations_scoring_log = investigations_scoring_log

  def evaluate_investigations_item(self, investigations_suggestions, investigations_item):
    """Evaluate investigations suggestion for one investigations item on checklist"""

    template = """
    You are an expert evaluator of doctors suggesting investigations for a patient.

    These are the investigations suggested by the doctor:
    {investigations_suggestions}

    Now make an assessment if the doctor suggested the appropriate investigations for '{investigations_item}'
    """

    prompt = PromptTemplate.from_template(template)

    structured_llm = self.model.with_structured_output(InvestigationsEvaluation)

    chain = prompt | structured_llm

    output = chain.invoke({
      "investigations_suggestions": investigations_suggestions,
      "investigations_item": investigations_item
    })

    return output

  def evaluate_ddx_checklist(self, ddx_suggestions, ddx_checklist):
    """Evaluate ddx suggestion with ddx checklist"""
    ddx_score = 0
    ddx_scoring_log = []
    for _, ddx_item in ddx_checklist.iterrows():
      output = self.evaluate_ddx_item(ddx_suggestions, ddx_item['Differential Diagnosis'])
      ddx_score += output.score
      ddx_scoring_log.append(output.explanation)

    self.differential_diagnoses_score = ddx_score
    self.differential_diagnoses_scoring_log = ddx_scoring_log

  def evaluate_ddx_item(self, ddx_suggestions, ddx_item):
    """Evaluate ddx suggestion for one ddx item on checklist"""

    template = """
    You are an expert evaluator of doctors providing differential diagnoses for a patient.

    These are the differential diagnoses suggested by the doctor:
    {ddx_suggestions}

    Now make an assessment if the doctor suggested the appropriate differential diagnoses for '{ddx_item}'
    """

    prompt = PromptTemplate.from_template(template)

    structured_llm = self.model.with_structured_output(DdxEvaluation)

    chain = prompt | structured_llm

    output = chain.invoke({
      "ddx_suggestions": ddx_suggestions,
      "ddx_item": ddx_item
    })

    return output
