# Supervisor-AI-Agent

## Overview

- This project implements a Technical Delivery Manager (TDM) agent using LangChain, designed to automate tasks related to video processing, transcript extraction, and summarization. Using a multi-agent workflow which consists of a supervisor that can supervise the work of multiple agents and return the work to the user

## Installation

### Prerequisites

- Python 3.8 or higher
- pip3

### Steps to Install

- Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

- Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```

-Install dependencies and export keys

```bash
pip3 install -r requirements.txt
export OPENAI_API_KEY=
export TAVILY_API_KEY=
```

## Usage

- Running the supervisor
- Ensure all prerequisites and dependencies are installed.
- Modify the .env file to include necessary environment variables.

```bash
python3 supervisor.py
```
