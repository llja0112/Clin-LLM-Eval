"""Index module containing multiple agent modules"""

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_community.llms import Replicate

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd

from simulator.retrievers import TemplateStore
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
    description="If a similar physical examination step is suggested \
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
    input_list = []

    for _, checklist_item in history_taking_checklist.iterrows():
      question = checklist_item['Question']
      patient_response = checklist_item['Patient Response']
      input_list.append({
        "conversation": conversation,
        "history_taking_category": question,
        "patient_response": patient_response
      })

    prompt = TemplateStore.get_prompt_template("history")

    structured_llm = self.model.with_structured_output(HistoryEvaluation)

    chain = prompt | structured_llm

    output = chain.batch(input_list, config={"max_concurrency": 10})

    return output

  def evaluate_physical_exam(self, physical_exam_suggestions, physical_exam_checklist):
    """Evaluate history taking capability of doctor based on checklist"""
    input_list = []

    for _, checklist_item in physical_exam_checklist.iterrows():
      physical_exam_step = checklist_item['Physical Exam']
      physical_exam_justification = checklist_item['Justification']

      input_list.append({
        "physical_exam_suggestions":physical_exam_suggestions,
        "physical_exam_step": physical_exam_step,
        "physical_exam_justification": physical_exam_justification
      })

    prompt = TemplateStore.get_prompt_template("physical")

    structured_llm = self.model.with_structured_output(PhysicalExamEvaluation)

    chain = prompt | structured_llm

    output = chain.batch(input_list, config={"max_concurrency": 10})

    return output

  def evaluate_investigations(self, investigations_suggestions, investigations_checklist):
    """Evaluate investigations suggestions with investigations checklist"""
    input_list = []

    for _, checklist_item in investigations_checklist.iterrows():
      investigations_item = checklist_item['Investigations']
      investigations_justification = checklist_item['Justification']

      input_list.append({
        "investigations_suggestions": investigations_suggestions,
        "investigations_item": investigations_item,
        "investigations_justification": investigations_justification
      })

    prompt = TemplateStore.get_prompt_template("investigations")

    structured_llm = self.model.with_structured_output(InvestigationsEvaluation)

    chain = prompt | structured_llm

    output = chain.batch(input_list, config={"max_concurrency": 10})

    return output

  def evaluate_ddx(self, ddx_suggestions, ddx_checklist):
    """Evaluate ddx suggestion with ddx checklist"""
    input_list = []

    for _, ddx_item in ddx_checklist.iterrows():
      diagnosis = ddx_item['Diagnosis']
      input_list.append({
        "ddx_suggestions": ddx_suggestions,
        "ddx_item": diagnosis,
      })

    prompt = TemplateStore.get_prompt_template("diagnosis")

    structured_llm = self.model.with_structured_output(DdxEvaluation)

    chain = prompt | structured_llm

    output = chain.batch(input_list, config={"max_concurrency": 10})

    return output
