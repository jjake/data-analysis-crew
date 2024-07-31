#!/usr/bin/env python
import sys

from crewai.crews import CrewOutput
from langchain_community.llms import ollama
from langchain_community.llms import openai
from zoocrew.crew import ZooCrew
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import dotenv
import os
from json import dumps

openai_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        model=os.getenv("OPENAI_MODEL_NAME"))

groq_llm = ChatGroq(
  api_key=os.getenv('GROQ_API_KEY'),
#  model="llama-3.1-70b-versatile",
  model="llama-3.1-8b-instant",
  max_tokens=2000,
  max_retries=1
)


ollama_llm = ollama.Ollama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

inputs = {
  'animals': 'lions, tigers, capybaras, tarantulas'
}

def run():

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (ZooCrew(llm=openai_llm).
                        feeding_crew().
                        kickoff(inputs=inputs))
  print(result.raw)
  return result
