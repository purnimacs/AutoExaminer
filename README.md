# AutoExaminer
AutoExaminer
AutoExaminer is an advanced question paper generator and evaluator that leverages AI to create and assess educational content. The system automates the creation of diverse question types from textbook content and offers evaluation capabilities for handwritten answer sheets.

📋 Table of Contents
Features
Architecture
Installation
Question Generation
Evaluation System
✨ Features
Question Generation
Generates four types of questions:
Multiple Choice Questions (MCQs)
Short Answer Questions
Long Answer Questions
Case Study Questions
Two difficulty levels:
Standard
Difficult
AI-driven topic selection based on textbook content
Vector-based content retrieval for contextually appropriate questions
Answer Evaluation
Evaluates handwritten answer sheets
Compares submissions against answer keys
Utilizes both LLM-based semantic understanding and pattern matching
Provides detailed feedback and scoring
🏗️ Architecture
AutoExaminer is built with the following components:

Content Processing Pipeline

Textbook content ingestion
Text chunking and preprocessing
Vector embeddings via LangChain
Storage in Chroma vector database
Question Generation Engine

Topic selection via LLM
Context retrieval from vector database
Question formulation using LLM with specialized prompts
Difficulty level adjustment
Evaluation System

Handwritten text processing
Answer key comparison
Pattern matching verification
LLM-based semantic evaluation
User Interface

Built using Appscrips
Integrated with Google Drive for data storage and retrieval
🔧 Installation
Prerequisites
Python 3.8+
Google Drive API access
AppScripts environment
LangChain and Chroma DB
Gemini API key
Setup
Clone the repository:
git clone https://github.com/EshaKhadeejaC/AutoExaminer.git
cd AutoExaminer
📝 Question Generation
AutoExaminer generates questions through the following process:

Content Analysis: The textbook is processed, chunked, and vectorized
Topic Selection: An LLM analyzes the content to identify key topics
Question Formulation: For each selected topic:
Relevant chunks are retrieved from the vector database
The LLM formulates questions based on the chunk content and question type
Difficulty level is applied to adjust complexity
Question Types
MCQs
Generated with one correct answer and three plausible distractors.

Short Answer
Designed to be answered in 2-3 sentences, focusing on definitions and concepts.

Long Answer
In-depth questions requiring detailed explanations, analysis, or discussion of topics.

Case Study
Real-world scenarios that require application of concepts from the textbook.

📊 Evaluation System
The evaluation system combines:

1.Handwritten Text Recognition: Processes scanned answer sheets
2.Pattern Matching: Checks for specific keywords and phrases from the answer key
3.Semantic Understanding: Uses LLM to evaluate the overall meaning and correctness
4.Scoring Algorithm: Weights different aspects of the answer to provide a fair assessment
💻 UI Integration
The UI is built using AppScripts and integrates with Google Drive for data management:

1.User inputs preferences in the AppScripts interface
2.Data is sent to Google Drive
3.Backend processes the request and stores results in Google Drive
4.AppScripts UI retrieves and displays the results
