#!/usr/bin/env python
import sys

from crewai.crews import CrewOutput
from langchain_community.llms import ollama
from langchain_community.llms import openai
from data_analyst.crew import DataAnalysisCrew
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import dotenv
import os
from json import dumps
import argparse


def manage_args():
  def read_file(f):
    with open(f, 'r') as file:
      file_contents = file.read()
      return file_contents

  parser = argparse.ArgumentParser(description='Virtual data analytic crew')
  parser.add_argument("--dir",
                      type=str,
                      help="root directory for data analysis")
  args = parser.parse_args()
  args = parser.parse_args()
  root_dir = args.dir
  root = './data/' + root_dir + '/'
  return {
    'root': root,
    'data_card': root + '/data_card.txt',
    'data_dictionary': root + '/data_dictionary/',
    'analytics_root': root + '/analytical_results/'
  }


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
  model="llama3.1",
  base_url="http://localhost:11434")


def prepare_metadata():
  files = manage_args()

  inputs = {
    'data_card': files['data_card'],
    'metadata_dir': files['data_dictionary']
  }

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (DataAnalysisCrew(llm=openai_llm).
                        prepare_metadata_crew().
                        kickoff(inputs=inputs))
  print(result.raw)
  with open(files['analytics_root'] + '/data_dictionary.json', 'w') as out_file:
    out_file.write(result.raw)
    out_file.close()

def formulate_hypotheses():
  files = manage_args()

  inputs = {
    'business_objective' : 'Identify the attributes of customers with a low risk of default and suggest growth programs to attract more customers like them',
    'data_dictionary' : files['analytics_root'] + '/data_dictionary.json',
    'num_hypotheses' : '10'
  }

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (DataAnalysisCrew(llm=openai_llm).
                        formulate_hypotheses_crew().
                        kickoff(inputs=inputs))
  print(result.raw)
  with open(files['analytics_root'] + '/hypotheses.txt', 'w') as out_file:
    out_file.write(result.raw)
    out_file.close()


def attach_statistical_tests():
  files = manage_args()
  hypotheses_file = files['analytics_root'] + '/hypotheses.txt'
  stat_tests_file = files['analytics_root'] + '/hypotheses_with_tests.txt'

  inputs = {
    'hypotheses_file' : hypotheses_file,
    'data_dictionary' : files['analytics_root'] + '/data_dictionary.json',
  }

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (DataAnalysisCrew(llm=openai_llm).
                        attach_statistical_tests_crew().
                        kickoff(inputs=inputs))
  print(result.raw)
  with open(stat_tests_file, 'w') as out_file:
    out_file.write(result.raw)
    out_file.close()

def attach_code():
  files = manage_args()
  stat_tests_file = files['analytics_root'] + '/hypotheses_with_tests.txt'
  code_file = files['analytics_root'] + '/hypotheses_with_tests_and_code.txt'

  inputs = {
    'stats_file' : stat_tests_file,
    'data_dictionary' : files['analytics_root'] + '/data_dictionary.json',
    'business_objective': 'Identify the attributes of customers with a low risk of default and suggest growth programs to attract more customers like them'
  }

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (DataAnalysisCrew(llm=openai_llm).
                        attach_code_crew().
                        kickoff(inputs=inputs))
  print(result.raw)
  with open(code_file, 'w') as out_file:
    out_file.write(result.raw)
    out_file.close()
