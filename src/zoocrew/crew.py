from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, TXTSearchTool
from pydantic import BaseModel, Field
from typing import List, Optional

class AnimalFood(BaseModel):
    animals: str = Field(...,description="Animals to be fed")
    food: list[str] = Field(...,description="List of food to feed each animal")


@CrewBase
class ZooCrew():

    def __init__(self,llm):
        self.llm = llm

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def zookeeper(self) -> Agent:
        return Agent(
            config=self.agents_config['zookeeper'],
            tools=[SerperDevTool(),
                   ScrapeWebsiteTool(),
                   TXTSearchTool(txt="./data/food_inventory.txt")],
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            output_json=AnimalFood
        )

    @task
    def feed_animals_task(self) -> Task:
        return Task(
            config=self.tasks_config['feed_animals_task'],
            agent=self.zookeeper(),
            memory=True
        )

    @crew
    def feeding_crew(self) -> Crew:
        """Create the zookeepers!"""

        return Crew(
            agents=[self.zookeeper()],
            tasks=[self.feed_animals_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

