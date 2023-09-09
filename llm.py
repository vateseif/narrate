from typing import List
from core import AbstractLLM, AbstractLLMConfig

from pydantic import BaseModel, Field
from langchain.schema import HumanMessage
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser



class Plan(BaseModel):
  tasks: List[str] = Field(description="list of all tasks that the robot has to carry out")
  
class Objective(BaseModel):
  objective: str = Field(description="objective function to be applied to MPC")

class Optimization(BaseModel):
  objective: str = Field(description="objective function to be applied to MPC")
  constraints: List[str] = Field(description="constraints to be applied to MPC")


ParsingModel = {
  "plan": Plan,
  "objective": Objective,
  "optimization": Optimization
}

class BaseLLM(AbstractLLM):

  def __init__(self, cfg: AbstractLLMConfig) -> None:
    super().__init__(cfg)
    # init parser
    self.parser = PydanticOutputParser(pydantic_object=ParsingModel[self.cfg.parsing])
    # init prompt
    system_prompt = SystemMessagePromptTemplate.from_template(self.cfg.prompt)
    self.messages = [system_prompt.format(format_instructions=self.parser.get_format_instructions())]    
    

  def run(self, user_message:str) -> str:
    self.messages.append(HumanMessage(content=user_message))
    model_message = self.model(self.messages)
    self.messages.append(model_message)
    print(f"\33[92m {model_message.content} \033[0m \n")
    return self.parser.parse(model_message.content)
