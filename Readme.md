# Disha AI Career Advisor

## Overview
Disha is an AI-powered platform for Indian students, providing personalized career recommendations and skill development pathways. Uses Flask backend with LangGraph and Google Gemini, and a React frontend for a modern UI/UX.

## Features
- **Sequential Agent Flow**: Profile -> Career -> Skills -> Market -> Chat agents.
- **Structured Output**: JSON responses with profile, careers, skills, market insights, and summary.
- **React Frontend**: Interactive UI with tabs, collapsible cards, and clickable links using Material-UI and Tailwind CSS.
- **Flask Backend**: Serves API endpoints and React app.

## Requirements
- **Backend**:
  - Python 3.8+
  - Libraries: `langchain`, `langgraph>=0.2.0`, `langchain-google-genai`, `flask`
  - Google API Key for Gemini
- **Frontend**:
  - No installation needed (uses CDNs for React, Material-UI, Tailwind, Axios)
  - Browser supporting modern JavaScript

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd disha-ai