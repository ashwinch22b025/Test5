import json
import requests
import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("GitHub Repository Evaluation")

# Input username
userName = st.text_input("Enter GitHub Username")

# Agent class for Generative AI interaction
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append(SystemMessage(content=self.system))
    
    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.messages.append(AIMessage(content=result))
        return result
    
    def execute(self):
        chat = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
        result = chat.invoke(self.messages)
        return result.content

# Evaluation function
def evaluate_repository_with_gemini(repo_data, pr_data, commit_data, readme_content):
    """
    Evaluate a GitHub repository using the Gemini API based on a 5-point rubric.

    Args:
        repo_data (dict): Basic details about the repository.
        pr_data (dict): Pull request statistics.
        commit_data (dict): Commit statistics.
        readme_content (str): Content of the README file.

    Returns:
        str: Evaluation result for the repository.
    """
    rubric = [
        "Number of pull requests accepted",
        "Frequency of commits",
        "Number of forks",
        "Quality of README file",
        "Number of contributors",
        "Diversity of programming languages used"
    ]
    
    # Define the prompt
    prompt = f"""
    Evaluate the GitHub repository '{repo_data['name']}' using the following data:
    Repository Details:
    - Name: {repo_data['name']}
    - Description: {repo_data.get('description', 'No description')}
    - Language: {repo_data.get('language', 'Unknown')}
    - Forks: {repo_data.get('forks', 0)}
    - Open Issues: {repo_data.get('open_issues', 0)}
    
    Pull Request Data:
    - Open PRs: {pr_data.get('open_prs', 0)}
    - Merged PRs: {pr_data.get('merged_prs', 0)}

    Commit Data:
    - Total Commits: {commit_data.get('total_commits', 0)}
    - Recent Commit Frequency: {commit_data.get('recent_frequency', 'No data')}

    README Analysis:
    {readme_content or 'No README available'}

    Using the rubric below, score each criterion out of 5 marks:
    """ + "\n".join([f"{i+1}. {point}" for i, point in enumerate(rubric)]) + "\nProvide a detailed evaluation and the final score out of 25."
    
    # Initialize the Agent
    bot = Agent("Evaluate a GitHub repository using a rubric.")
    result = bot(prompt)
    return result

if userName:
    st.subheader("Repository Evaluations")

    # Fetch repository data
    repo_url = f'https://api.github.com/users/{userName}/repos'
    repos = requests.get(repo_url).json()

    if isinstance(repos, list):
        evaluations = []  # Store evaluations for each repository

        for repo in repos:
            # Skip intermediate printing, just collect data
            repo_data = {
                'name': repo.get('name', 'Unknown'),
                'description': repo.get('description', 'No description'),
                'language': repo.get('language', 'Unknown'),
                'forks': repo.get('forks', 0),
                'open_issues': repo.get('open_issues', 0),
            }

            # Pull Request Data
            pr_url = f'https://api.github.com/repos/{userName}/{repo["name"]}/pulls?state=all'
            pr_response = requests.get(pr_url).json()
            pr_data = {
                'open_prs': sum(1 for pr in pr_response if pr.get('state') == 'open'),
                'merged_prs': sum(1 for pr in pr_response if pr.get('state') == 'closed'),
            }

            # Commit Data
            commits_url = f'https://api.github.com/repos/{userName}/{repo["name"]}/commits'
            commit_response = requests.get(commits_url).json()
            commit_data = {
                'total_commits': len(commit_response) if isinstance(commit_response, list) else 0,
                'recent_frequency': "High" if len(commit_response) > 10 else "Low",
            }

            # README Data
            readme_url = f'https://api.github.com/repos/{userName}/{repo["name"]}/readme'
            readme_response = requests.get(readme_url).json()
            readme_content = readme_response.get('content', None)

            # Evaluate the repository
            evaluation_result = evaluate_repository_with_gemini(repo_data, pr_data, commit_data, readme_content)
            evaluations.append((repo_data['name'], evaluation_result))

        # Display results for each repository
        for idx, (repo_name, evaluation) in enumerate(evaluations, start=1):
            st.subheader(f"Repository {idx}: {repo_name}")
            st.write(evaluation)
    else:
        st.write("No repositories found or an error occurred.")
