Salbot - Your Personal AI Career Advisor
Salbot is a comprehensive, conversational AI-powered web application designed to provide personalized career guidance to students and recent graduates in India. It goes beyond simple recommendations by offering a full suite of tools to help users understand their path, prepare for it, and get a glimpse into the job market.

✨ Features
This application is built on a sophisticated agentic architecture, where multiple specialized AI agents work together to provide a holistic analysis.

🤖 Conversational Interface: Engage in natural, multi-turn conversations. The AI remembers your session's context for relevant follow-up questions.

🧠 Intelligent Routing: The app intelligently detects if you're asking a follow-up question or exploring a brand new career path, automatically triggering a fresh analysis when needed.

📄 PDF Resume Analysis: Upload your resume in PDF format for personalized feedback and suggestions tailored to your desired career paths.

🚀 Parallel Processing for Speed: Multiple AI agents run their analyses simultaneously to deliver a comprehensive report much faster than a sequential process.

Career Toolkit:

Personalized Recommendations: Get career suggestions based on your interests, skills, and background.

Skills Gap Analysis: Understand the skills you need to acquire for your target roles.

Custom Learning Roadmaps: Receive a step-by-step learning plan with links to resources.

Verified Resource Links: All learning resource links are generated as guaranteed-to-work Google search links to prevent 404 errors.

Market Insights: Get data on salary expectations and job market trends in India.

Interview Prep: Generate common technical and behavioral interview questions for each career.

Simulated Job Board: View realistic, AI-generated job and internship postings.

📥 PDF Export: Download your complete, structured career analysis as a PDF to review offline.

🛠️ Tech Stack
Backend: Python, Flask, LangChain, LangGraph

AI Model: Google Gemini API

Frontend: HTML, Tailwind CSS, JavaScript

Deployment: Gunicorn, Render

⚙️ Project Architecture
The application's backend is built using LangGraph, which orchestrates a graph of specialized AI agents.

Router Agent: The entry point that first analyzes the user's query to determine if it's a new analysis or a follow-up.

Profile & Career Agents: These run sequentially to build a user profile and generate initial career recommendations.

Parallel Analysis: Five agents then run simultaneously to gather comprehensive data:

Skills & Learning Roadmap Agent

Market Insights Agent

Resume Feedback Agent (only runs if a PDF was uploaded)

Interview Prep Agent

Job Board Agent

Link Generation: A special agent takes the learning resources and converts them into valid search URLs.

Collector & Final Response: The results from all parallel branches are collected and compiled into a final, structured JSON response for the user interface.

[User Query] -> [Router] --(new)--> [Profile] -> [Career] --+--> [Skills] -> [Link Gen] --+--> [Collector] -> [Final Response]
                          |                         |--> [Market] ---------|
                          |                         |--> [Resume] ---------|
                          '--(follow-up)--> [Chat]  |--> [Interview] ------|
                                                    '--> [Job Board] ------'

🚀 Getting Started
Prerequisites
Python 3.8+

pip package manager

A Google AI (Gemini) API Key

Local Development Setup
Clone the repository:

git clone <your-repository-url>
cd <your-repository-folder>

Create and activate a virtual environment:

On Windows:

python -m venv venv
.\venv\Scripts\activate

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Set up your environment variables:

Create a file named .env in the root of your project folder.

Add your Google API key to this file:

GOOGLE_API_KEY="your_actual_google_api_key_here"

Run the application:

python app.py

Open your browser and navigate to http://127.0.0.1:5000.

☁️ Deployment to Render
This project is configured for easy deployment on Render's free tier.

Push to GitHub: Ensure your project, including the requirements.txt, Procfile, and .gitignore files, is pushed to a public GitHub repository. Do not commit your .env file.

Create a New Web Service on Render:

Connect your GitHub account to Render.

Create a New Web Service and select your repository.

Render will automatically use the requirements.txt to install dependencies and the Procfile to set the start command (gunicorn app:app).

Add Environment Variable:

In the Render dashboard for your service, go to the "Environment" tab.

Add an environment variable with the Key GOOGLE_API_KEY and paste your secret key as the Value.

Deploy: Click "Create Web Service". Render will build and deploy your application, providing you with a public URL.

📁 File Structure
.
├── app.py              # Main Flask application, all backend logic and AI agents
├── index.html            # Single-page frontend UI
├── requirements.txt      # Python dependencies for pip
├── Procfile              # Start command for the deployment server (Gunicorn)
├── .env                  # Local environment variables (API key) - Not committed to Git
└── .gitignore            # Tells Git which files to ignore
