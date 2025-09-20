from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from dotenv import load_dotenv
import fitz  
import urllib.parse


load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    logging.warning("Google API Key not found in .env file.")
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY

llm = None
try:
    if API_KEY:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        logging.info("Gemini LLM initialized successfully.")
except Exception as e:
    logging.error(f"Failed during initialization: {e}")


def last_write_wins(x, y):
    return y

class SalbotState(TypedDict):
    student_profile: Annotated[Dict[str, Any], last_write_wins]
    career_recommendations: Annotated[List[Dict[str, Any]], last_write_wins]
    skills_data: Annotated[Dict[str, Any], last_write_wins]
    market_insights: Annotated[Dict[str, Any], last_write_wins]
    resume_feedback: Annotated[Dict[str, Any], last_write_wins]
    interview_prep: Annotated[Dict[str, Any], last_write_wins]
    job_postings: Annotated[Dict[str, Any], last_write_wins]
    conversation_history: Annotated[List[Dict[str, str]], last_write_wins]
    user_query: Annotated[str, last_write_wins]
    final_response: Annotated[Dict[str, Any], last_write_wins]
    intent: Annotated[str, last_write_wins]
    resume_uploaded: Annotated[bool, last_write_wins]


def invoke_llm(prompt: ChatPromptTemplate, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if not llm: return {"error": "LLM not initialized. Please check your API key."}
    try:
        chain = prompt | llm
        response = chain.invoke(inputs)
        content = response.content.strip().strip('```json').strip('```')
        return json.loads(content)
    except json.JSONDecodeError:
        return {"message": response.content}
    except Exception as e:
        logging.error(f"LLM invocation error: {e}")
        return {"error": str(e)}


router_system = "You are a routing agent. Determine the user's intent. If they ask about a new career, classify as 'new_analysis'. If they ask a related question, classify as 'follow_up'. Respond ONLY with JSON: {{\"intent\": \"follow_up\"}} or {{\"intent\": \"new_analysis\"}}"
router_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(router_system), HumanMessagePromptTemplate.from_template("History: {history}\n\nUser's new query: {query}")])

def router_agent(state: SalbotState) -> SalbotState:
    state.setdefault("conversation_history", []).append({"role": "user", "content": state["user_query"]})
    if len(state["conversation_history"]) <= 1:
        state["intent"] = "new_analysis"
        return state
    result = invoke_llm(router_prompt, {"history": json.dumps(state["conversation_history"]), "query": state["user_query"]})
    state["intent"] = result.get("intent", "follow_up")
    return state

def profile_agent(state: SalbotState) -> SalbotState:
    profile_system = "You are the Profile Builder. Analyze the user's input (including any resume text) to create a student profile. Output JSON: {{\"interests\": [...], \"skills\": {{...}}, ...}}"
    profile_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(profile_system), HumanMessagePromptTemplate.from_template("User input: {user_query}")])
    profile = invoke_llm(profile_prompt, {"user_query": state["user_query"]})
    state["student_profile"] = profile
    return state

def career_agent(state: SalbotState):
    career_system = "You are the Career Recommender. Match careers to the student's profile for the Indian job market. Output JSON: {{\"careers\": [{{\"career\": \"Name\", \"rationale\": \"Why\", ...}}]}}"
    career_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(career_system), HumanMessagePromptTemplate.from_template("Student profile: {profile}")])
    if not state.get("student_profile") or state["student_profile"].get("error"):
        state["career_recommendations"] = []
        return state
    recommendations_data = invoke_llm(career_prompt, {"profile": json.dumps(state["student_profile"])})
    state["career_recommendations"] = recommendations_data.get("careers", [])
    return state

def skills_agent_parallel(state: SalbotState):
    skills_system = "You are the Skills Analyzer. For each recommended career, identify skill gaps and generate learning roadmaps with suggested resources. DO NOT MAKE UP URLs. For resources, just name the course and platform (e.g., 'Python for Everybody on Coursera'). Output JSON: {{ \"skills\": {{\"career1\": {{\"gaps\": [...], \"roadmap\": {{\"steps\": [{{\"step\": \"Learn Python\", \"resources\": [\"Python for Everybody on Coursera\"], ...}}]}} }}}}}}"
    skills_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(skills_system), HumanMessagePromptTemplate.from_template("Profile: {profile}\nCareers: {careers}")])
    if not state.get("career_recommendations"): return state
    result = invoke_llm(skills_prompt, {"profile": json.dumps(state["student_profile"]), "careers": json.dumps(state["career_recommendations"])})
    state["skills_data"] = result.get("skills", {})
    return state

def link_generation_agent(state: SalbotState) -> SalbotState:
    logging.info("Generating search links for resources.")
    skills_data = state.get("skills_data", {})
    for career, details in skills_data.items():
        if "roadmap" in details and "steps" in details["roadmap"]:
            for step in details["roadmap"]["steps"]:
                generated_links = []
                for resource_name in step.get("resources", []):
                    query = urllib.parse.quote_plus(resource_name)
                    search_url = f"https://www.google.com/search?q={query}"
                    generated_links.append(search_url)
                step["resources"] = generated_links
    return state

def market_agent_parallel(state: SalbotState):
    market_system = "You are the Market Insights Provider. Summarize job data for India for the given careers. Output JSON: {{\"insights\": {{\"career1\": {{\"salary\": \"...\", ...}}}}}}"
    market_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(market_system), HumanMessagePromptTemplate.from_template("Careers: {careers}")])
    if not state.get("career_recommendations"): return state
    insights_data = invoke_llm(market_prompt, {"careers": json.dumps(state["career_recommendations"])})
    state["market_insights"] = insights_data.get("insights", {})
    return state

def resume_agent_parallel(state: SalbotState) -> SalbotState:
    if not state.get("resume_uploaded"):
        logging.info("Skipping resume agent as no resume was uploaded.")
        state["resume_feedback"] = {"feedback_summary": "No resume was uploaded for this analysis.", "suggestions": []}
        return state

    resume_system = "You are a Resume Advisor. The user has uploaded their resume. Analyze the resume text provided in their query, compare it to the skills required for the recommended careers, and provide specific, actionable feedback. Output JSON: {{\"feedback_summary\": \"...\", \"suggestions\": [...]}}."
    resume_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(resume_system), HumanMessagePromptTemplate.from_template("User's query with resume text: {user_query}\nRecommended careers: {careers}")])
    
    if not state.get("career_recommendations"): 
        state["resume_feedback"] = {}
        return state

    feedback = invoke_llm(resume_prompt, {"user_query": state["user_query"], "careers": json.dumps(state["career_recommendations"])})
    state["resume_feedback"] = feedback
    return state

def interview_agent_parallel(state: SalbotState) -> SalbotState:
    interview_system = "You are an Interview Prep Coach. Generate technical and behavioral questions for each recommended career. Output JSON: {{\"interview_prep\": {{\"career1\": {{\"technical\": [...], \"behavioral\": [...]}}}}}}"
    interview_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(interview_system), HumanMessagePromptTemplate.from_template("Recommended careers: {careers}")])
    if not state.get("career_recommendations"): return state
    prep_data = invoke_llm(interview_prompt, {"careers": json.dumps(state["career_recommendations"])})
    state["interview_prep"] = prep_data.get("interview_prep", {})
    return state

def job_board_agent_parallel(state: SalbotState) -> SalbotState:
    job_board_system = "You are a Simulated Job Board. Create 2-3 FAKE job postings for each recommended career in India. Output JSON: {{\"postings\": {{\"career1\": [{{\"title\": \"...\", ...}}, ...]}}}}"
    job_board_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(job_board_system), HumanMessagePromptTemplate.from_template("Recommended careers: {careers}")])
    if not state.get("career_recommendations"): return state
    postings_data = invoke_llm(job_board_prompt, {"careers": json.dumps(state["career_recommendations"])})
    state["job_postings"] = postings_data.get("postings", {})
    return state

def collect_results_agent(state: SalbotState) -> SalbotState:
    logging.info("All parallel analyses have completed. Collecting results.")
    return state

def initial_response_agent(state: SalbotState):
    initial_response_system = "You are Salbot. Combine all analysis data into a structured JSON for the UI. Write a friendly, encouraging summary message. Output JSON: {{\"profile_summary\": \"...\", \"careers\": [...], \"skills\": {{...}}, \"market_insights\": {{...}}, \"resume_feedback\": {{...}}, \"interview_prep\": {{...}}, \"job_postings\": {{...}}, \"message\": \"...\"}}"
    initial_response_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(initial_response_system), HumanMessagePromptTemplate.from_template("Full State: {state}")])
    
    skills_for_prompt = {career_name: {"gaps": data.get("gaps", []), "roadmap": data.get("roadmap", {})} for career_name, data in state.get("skills_data", {}).items()}
    full_state_for_prompt = { "profile": state.get("student_profile", {}), "careers": state.get("career_recommendations", []), "skills": skills_for_prompt, "market_insights": state.get("market_insights", {}), "resume_feedback": state.get("resume_feedback", {}), "interview_prep": state.get("interview_prep", {}), "job_postings": state.get("job_postings", {}), }
    response = invoke_llm(initial_response_prompt, {"state": json.dumps(full_state_for_prompt)})
    response.setdefault("profile_summary", "Profile analysis complete.")
    response.setdefault("careers", state.get("career_recommendations", []))
    response.setdefault("skills", skills_for_prompt)
    response.setdefault("market_insights", state.get("market_insights", {}))
    response.setdefault("resume_feedback", state.get("resume_feedback", {"feedback_summary": "Not analyzed.", "suggestions": []}))
    response.setdefault("interview_prep", state.get("interview_prep", {}))
    response.setdefault("job_postings", state.get("job_postings", {}))
    response.setdefault("message", "Here is your comprehensive career toolkit!")
    state["final_response"] = response
    state["conversation_history"].append({"role": "assistant", "content": response["message"]})
    return state

def follow_up_agent(state: SalbotState):
    follow_up_system = "You are Salbot, an AI career advisor. Answer the user's follow-up question based on the provided conversation history and the full analysis data. Be conversational. Do not output JSON."
    follow_up_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(follow_up_system), HumanMessagePromptTemplate.from_template("History: {history}\n\nFull Analysis Data: {full_analysis}\n\nUser's new question: {query}")])
    response = invoke_llm(follow_up_prompt, {"history": json.dumps(state.get("conversation_history", [])), "full_analysis": json.dumps(state.get("final_response", {})), "query": state["user_query"]})
    final_response_package = state.get("final_response", {})
    final_response_package["message"] = response.get("message", "I'm not sure how to answer that.")
    state["final_response"] = final_response_package
    state["conversation_history"].append({"role": "assistant", "content": final_response_package["message"]})
    return state

workflow = StateGraph(SalbotState)
workflow.add_node("router_agent", router_agent)
workflow.add_node("profile_agent", profile_agent)
workflow.add_node("career_agent", career_agent)
workflow.add_node("skills_agent", skills_agent_parallel)
workflow.add_node("link_generator", link_generation_agent) 
workflow.add_node("market_agent", market_agent_parallel)
workflow.add_node("resume_agent", resume_agent_parallel)
workflow.add_node("interview_agent", interview_agent_parallel)
workflow.add_node("job_board_agent", job_board_agent_parallel)
workflow.add_node("collect_results", collect_results_agent)
workflow.add_node("initial_response_agent", initial_response_agent)
workflow.add_node("follow_up_agent", follow_up_agent)

def decide_next_step(state: SalbotState):
    return "profile_agent" if state.get("intent") == "new_analysis" else "follow_up_agent"

workflow.set_entry_point("router_agent")
workflow.add_conditional_edges("router_agent", decide_next_step, {"profile_agent": "profile_agent", "follow_up_agent": "follow_up_agent"})
workflow.add_edge("profile_agent", "career_agent")
workflow.add_edge("career_agent", "skills_agent")
workflow.add_edge("skills_agent", "link_generator") 
workflow.add_edge("link_generator", "collect_results") 
workflow.add_edge("career_agent", "market_agent")
workflow.add_edge("career_agent", "resume_agent")
workflow.add_edge("career_agent", "interview_agent")
workflow.add_edge("career_agent", "job_board_agent")
workflow.add_edge("market_agent", "collect_results")
workflow.add_edge("resume_agent", "collect_results")
workflow.add_edge("interview_agent", "collect_results")
workflow.add_edge("job_board_agent", "collect_results")
workflow.add_edge("collect_results", "initial_response_agent")
workflow.add_edge("initial_response_agent", END)
workflow.add_edge("follow_up_agent", END)

checkpointer = MemorySaver()
langgraph_app = workflow.compile(checkpointer=checkpointer)

@app.route('/api/query', methods=['POST'])
def handle_query():
    if not llm: return jsonify({"error": "LLM not initialized. Please check your API key."}), 500
    
    full_query = ""
    thread_id = 'default_student'
    resume_uploaded_flag = False

    if 'resume_pdf' in request.files:
        resume_uploaded_flag = True
        pdf_file = request.files['resume_pdf']
        text_query = request.form.get('query', '')
        thread_id = request.form.get('thread_id', 'default_student')
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            resume_text = "".join(page.get_text() for page in doc)
            full_query = f"--- RESUME CONTENT ---\n{resume_text}\n--- END RESUME ---\n\n{text_query}"
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            return jsonify({"error": f"Failed to process PDF file: {e}"}), 400
    else:
        data = request.get_json()
        if not data or 'query' not in data: return jsonify({"error": "Missing 'query'"}), 400
        full_query = data.get('query')
        thread_id = data.get('thread_id', 'default_student')

    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        initial_state = {"user_query": full_query, "resume_uploaded": resume_uploaded_flag}
        result = langgraph_app.invoke(initial_state, config=config)
        final_response = result.get("final_response", {"error": "No response generated."})
        if isinstance(final_response, dict) and "error" in final_response:
             return jsonify(final_response), 500
        return jsonify(final_response)
    except Exception as e:
        logging.error(f"Error processing graph: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

