from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import io
import os
import PyPDF2
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

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
        # Use Gemini API without specifying the version
        chat = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,convert_system_message_to_human=True)
        result = chat.invoke(self.messages)
        return result.content

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

# Function to extract structured data from CV text
def extract_structured_data(cv_text):
    structured_prompt_2 = """ 
Read the following cv_text deeply and convert it into a JSON data structure with the following specifications:

1. Each key should be a category relevant to the CV, covering all available sections. Some common keys include "Skills", "Title", "Education", "Work Experience", "Languages", "Bio", "Hourly Rate", "Phone Number", "Date of Birth", "Address", "Country", "State", "City/Province", "Zip/Postal Code", "Achievements", "Certifications", "Extracurricular Activities", "Projects", "Por", and "Publications". 
2. Every value must be an **array of strings**, where:
   - For text-based descriptions, break them into meaningful sentences or points as elements of the array.
   - For fields like education or work experience, if structured data is needed, keep them as arrays of dictionaries, with each field containing arrays of strings.
3. Extract as much structured and meaningful information as possible, filling all keys that are present in the CV. Use empty arrays for values not mentioned.
4. Provide a comprehensive output that leaves no detail from the CV unaccounted for create your own key value pair for anything which doesnot come under the below category but everything in the cv should be covered.
   
Example structure for the output:
{
  "Skills": ["Skill1", "Skill2", ...],
  "Title": ["Primary job title or profession"],
  "Education": [
    {
      "Institution": ["Institution Name"],
      "Degree": ["Degree Type"],
      "Field of Study": ["Field of Study"],
      "Start Date": ["Start Date"],
      "End Date": ["End Date"],
      "Description": ["Description about the education"]
    },
    ...
  ],
  "Work Experience": [
    {
      "Company Name": ["Company Name"],
      "Position": ["Position"],
      "Start Date": ["Start Date"],
      "End Date": ["End Date"],
      "Description": ["Description of the role"]
    },
    ...
  ],
  "Achievements": ["Achievement1", "Achievement2", ...],
  "Certifications": ["Certification1", "Certification2", ...],
  "Extracurricular Activities": ["Activity1", "Activity2", ...],
  "Projects": [
    {
      "Title": ["Project Title"],
      "Description": ["Description about the project"],
      "Technologies Used": ["Technology1", "Technology2", ...]
    },
    ...
  ],
  "Languages": [
    {
      "Language": ["Language Name"],
      "Proficiency": ["Proficiency Level (basic, fluent, expert)"]
    },
    ...
  ],
  "Por": ["Description about the Position of responsibilities if mentioned"],
  "Publications": ["Publication1", "Publication2", ...],
  "Bio": ["Summary paragraph about the individual"],
  "Hourly Rate": ["Rate if mentioned"],
  "Phone Number": ["Phone Number if mentioned"],
  "Date of Birth": ["Date of Birth if mentioned"],
  "Address": ["Address if mentioned"],
  "Country": ["Country if mentioned"],
  "State": ["State if mentioned"],
  "City/Province": ["City/Province if mentioned"],
  "Zip/Postal Code": ["Zip or Postal Code if mentioned"]
}
Use empty array for values not mentioned. Do not use null, use empty arrays if no values are found.
The output should cover **everything from the CV using key value pairs it should not miss anything from the cv_text provided make sure it uses every information from cv_text** comprehensively and use arrays to ensure uniformity.

Explanation of each key and format requirements:
Skills:
List of technical, professional, or soft skills relevant to the individual.
Title:
The main job title or profession being sought or described.
Education:
An array of dictionaries containing information on the educational qualifications like institution name, degree type, field of study, and duration.
Work Experience:
An array of dictionaries detailing roles, responsibilities, and work periods at various companies or organizations.
Achievements:
A list of significant accomplishments like awards, recognitions, competitions, scholarships, etc.
Certifications:
A list of certifications, courses, or training programs undertaken by the individual.
Extracurricular Activities:
A list of activities, projects, hobbies, or other engagements outside of formal education and work.
Projects:
An array of dictionaries detailing personal, academic, or professional projects, including titles, descriptions, and technologies used.
Languages:
An array of dictionaries specifying languages spoken by the individual along with proficiency levels (e.g., basic, fluent, expert).
Por:
A list describing positions of responsibility held, particularly roles that required leadership, teamwork, or special duties.
Publications:
A list of research papers, articles, books, or other published works authored by the individual.
Bio:
A paragraph summarizing the individual's background, key strengths, and career aspirations.
Hourly Rate:
The individual's expected or mentioned hourly rate, if applicable.
Phone Number:
The individual's contact number, if provided.
Date of Birth:
The individual's date of birth, if provided.
Address:
The individual's address, if mentioned.
Country:
The country of residence, if mentioned.
State:
The state or province of residence, if mentioned.
City/Province:
The city or province of residence, if mentioned.
Zip/Postal Code:
The zip or postal code, if mentioned.
""".strip() 
    bot2 = Agent(structured_prompt_2)
    res = bot2(cv_text)
    text_to = res.strip().strip('```').strip('```json')
    return json.loads(text_to)

def save_json_to_file(data, filename="output.json"):
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving JSON file: {str(e)}")

# FastAPI route for processing PDF and extracting structured data
@app.post("/process_resume")
async def process_resume(file: UploadFile = File(...)):
    try:
        # Step 1: Read the PDF file
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        # Step 2: Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_file)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
        # Step 3: Extract structured data from the text
        structured_data = extract_structured_data(extracted_text)
        save_json_to_file(structured_data)
        return {
            "text": extracted_text,
            "structured_data": structured_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Resume Processing API!",
        "instructions": "Use /process_resume to upload a PDF and extract structured data."
    }