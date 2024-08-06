from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (SerperDevTool,
                          ScrapeWebsiteTool,
                          TXTSearchTool,
                          DirectorySearchTool,
                          DirectoryReadTool,
                          CSVSearchTool,
                          FileReadTool)
from pydantic import BaseModel, Field
from typing import List, Optional

class AnimalFood(BaseModel):
    animals: str = Field(...,description="Animals to be fed")
    food: list[str] = Field(...,description="List of food to feed each animal")


@CrewBase
class DataAnalysisCrew():

    def __init__(self,llm):
        self.llm = llm

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
        )

    @agent
    def business_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['business_analyst'],
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
        )

    @agent
    def statistician(self) -> Agent:
        return Agent(
            config=self.agents_config['statistician'],
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
        )
    @task
    def build_data_metadata_task(self) -> Task:
        return Task(
            config=self.tasks_config['build_data_metadata_task'],
            agent=self.data_analyst(),
            tools=[DirectoryReadTool(),FileReadTool()],
            memory=True
        )

    @task
    def contextual_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['contextual_research_task'],
            agent=self.data_analyst(),
            tools=[DirectoryReadTool(),
                   FileReadTool(),
                   SerperDevTool(),
                   TXTSearchTool()],
            memory=True
        )
    @task
    def formulate_hypotheses_task(self) -> Task:
        return Task(
            config=self.tasks_config['formulate_hypotheses_task'],
            agent=self.business_analyst(),
            tools=[FileReadTool()],
            memory=True
        )
    @task
    def attach_statistical_tests_task(self) -> Task:
        return Task(
            config=self.tasks_config['attach_statistical_tests_task'],
            agent=self.statistician(),
            tools=[FileReadTool()],
            memory=True
        )
    @task
    def attach_code_task(self) -> Task:
        return Task(
            config=self.tasks_config['attach_code_task'],
            agent=self.data_scientist(),
            tools=[FileReadTool()],
            memory=True
        )

    @agent
    def data_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['data_scientist'],
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
        )
    @crew
    def prepare_metadata_crew(self) -> Crew:

        return Crew(
            agents=[self.data_analyst()],
            tasks=[self.build_data_metadata_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

    def formulate_hypotheses_crew(self) -> Crew:
        return Crew(
            agents=[self.data_analyst()],
            tasks=[self.contextual_research_task(),
                   self.formulate_hypotheses_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )
    def attach_statistical_tests_crew(self) -> Crew:
        return Crew(
            agents=[self.statistician()],
            tasks=[self.attach_statistical_tests_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

    def attach_code_crew(self) -> Crew:
        return Crew(
            agents=[self.data_scientist()],
            tasks=[self.attach_code_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )


