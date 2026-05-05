from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="FlightMind AI - Agentic Flight Operations Assistant")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Sample flight data for RAG
FLIGHT_DATA = """
Flight AA101: New York JFK to Los Angeles LAX. Departure 08:00, Arrival 11:30.
Status: On Time. Gate B12. Aircraft: Boeing 737.

Flight AA202: Chicago ORD to Dallas DFW. Departure 14:00, Arrival 16:45.
Status: Delayed 30 minutes. Gate C8. Aircraft: Airbus A320.

Flight AA303: Miami MIA to Boston BOS. Departure 09:15, Arrival 13:00.
Status: Cancelled due to weather. Rebooking available.

Flight AA404: Seattle SEA to Denver DEN. Departure 17:30, Arrival 20:45.
Status: On Time. Gate A5. Aircraft: Boeing 757.

Flight AA505: San Francisco SFO to New York JFK. Departure 07:00, Arrival 15:30.
Status: On Time. Gate D3. Aircraft: Boeing 777.

Flight AA606: Los Angeles LAX to Chicago ORD. Departure 11:00, Arrival 17:00.
Status: Delayed 15 minutes. Gate E7. Aircraft: Airbus A321.

Baggage Policy: First bag free for AAdvantage members.
Second bag $40. Overweight bags over 50lbs cost $100.
Carry-on: One free carry-on plus personal item allowed.
International flights: First checked bag $75 for non-members.

Loyalty Program: AAdvantage miles earned at 5 miles per dollar spent.
Elite status tiers: Gold (25K miles), Platinum (50K), Executive Platinum (100K).
Gold members earn 7 miles per dollar. Platinum earn 8 miles per dollar.
Executive Platinum earn 11 miles per dollar spent.

Rebooking Policy: Cancelled flights qualify for full refund or free rebooking.
Delayed flights over 3 hours qualify for meal vouchers worth $15.
Same-day flight changes available for $75 fee for non-elite members.
Elite members get free same-day flight changes.

Check-in Policy: Online check-in opens 24 hours before departure.
Airport check-in closes 45 minutes before domestic flights.
Airport check-in closes 60 minutes before international flights.
"""

# Build RAG vectorstore
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    docs = splitter.create_documents([FLIGHT_DATA])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

vectorstore = build_vectorstore()

@tool
def search_flight_info(query: str) -> str:
    """Search for flight information, status, baggage policy, check-in rules, and loyalty program details."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

@tool
def get_current_time() -> str:
    """Get the current date and time for flight scheduling context."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_miles(spend_amount: float, elite_tier: str = "none") -> str:
    """
    Calculate AAdvantage miles earned for a given spend amount.
    elite_tier options: none, gold, platinum, executive_platinum
    """
    rates = {
        "none": 5,
        "gold": 7,
        "platinum": 8,
        "executive_platinum": 11
    }
    rate = rates.get(elite_tier.lower(), 5)
    miles = int(spend_amount * rate)
    return (
        f"Miles calculation for ${spend_amount:.2f} spend:\n"
        f"Elite Tier: {elite_tier.title()}\n"
        f"Rate: {rate} miles per dollar\n"
        f"Total Miles Earned: {miles:,} AAdvantage miles"
    )

@tool
def check_rebooking_options(flight_number: str) -> str:
    """Check rebooking options for cancelled or delayed flights."""
    options = {
        "AA303": {
            "status": "Cancelled",
            "reason": "Weather",
            "options": [
                "Option 1: AA305 Miami to Boston - Departing 15:30 same day - Available",
                "Option 2: AA307 Miami to Boston - Departing 18:00 same day - Available",
                "Option 3: Full refund to original payment method within 7 business days",
                "Option 4: Travel credit for future use with 10% bonus"
            ]
        },
        "AA202": {
            "status": "Delayed 30 minutes",
            "reason": "Air traffic control",
            "options": [
                "Current flight AA202 will depart at 14:30 instead of 14:00",
                "Meal voucher of $15 not applicable (delay under 3 hours)",
                "Option to rebook on AA204 departing 16:00 at no charge"
            ]
        },
        "AA606": {
            "status": "Delayed 15 minutes",
            "reason": "Late incoming aircraft",
            "options": [
                "Current flight AA606 will depart at 11:15 instead of 11:00",
                "No rebooking necessary for this minor delay"
            ]
        }
    }

    flight_upper = flight_number.upper()
    if flight_upper in options:
        flight_info = options[flight_upper]
        result = f"Flight {flight_upper} Status: {flight_info['status']}\n"
        result += f"Reason: {flight_info['reason']}\n"
        result += "Available Options:\n"
        result += "\n".join(flight_info["options"])
        return result
    return f"Flight {flight_number} is operating normally with no disruptions. No rebooking needed."

@tool
def get_airport_info(airport_code: str) -> str:
    """Get information about airport terminals, gates, and services."""
    airports = {
        "JFK": "John F. Kennedy International Airport, New York. Terminals 1-8. American Airlines operates from Terminal 8. AirTrain connects all terminals.",
        "LAX": "Los Angeles International Airport. Terminals 1-8 plus Tom Bradley International. American Airlines in Terminals 4 and 5.",
        "ORD": "O'Hare International Airport, Chicago. Terminals 1-3 and 5. American Airlines in Terminal 3.",
        "DFW": "Dallas Fort Worth International Airport. Terminals A-E. American Airlines hub - operates all terminals.",
        "MIA": "Miami International Airport. Concourses A-H. American Airlines hub in Concourse D.",
        "BOS": "Boston Logan International Airport. Terminals A-E. American Airlines in Terminal B.",
        "SEA": "Seattle-Tacoma International Airport. Single main terminal with concourses A-N. American Airlines in Concourse B.",
        "DEN": "Denver International Airport. Jeppesen Terminal with Concourses A-C. American Airlines in Concourse B.",
        "SFO": "San Francisco International Airport. Terminals 1-3 and International. American Airlines in Terminal 2."
    }
    code_upper = airport_code.upper()
    if code_upper in airports:
        return airports[code_upper]
    return f"Airport information for {airport_code} not available. Please check aa.com for details."

# Build agent
tools = [
    search_flight_info,
    get_current_time,
    calculate_miles,
    check_rebooking_options,
    get_airport_info
]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are FlightMind AI, an intelligent and empathetic flight operations 
    assistant for American Airlines. You help passengers with:
    - Flight status and gate information
    - Baggage policies and fees
    - Loyalty program miles calculation
    - Rebooking options for disrupted flights
    - Airport information and navigation
    - Check-in policies and procedures
    
    Always be helpful, accurate, and empathetic especially when flights are cancelled or delayed.
    Use your tools to find accurate information before responding.
    If you cannot find specific information, direct passengers to aa.com or 1-800-433-7300."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10
)

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

class QueryRequest(BaseModel):
    message: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str

@app.get("/")
def root():
    return {
        "message": "FlightMind AI - Agentic Flight Operations Assistant",
        "status": "running",
        "version": "1.0.0",
        "tools_available": len(tools)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "gpt-4-turbo",
        "tools": [t.name for t in tools],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        result = agent_executor.invoke({"input": request.message})
        return QueryResponse(
            response=result["output"],
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flights")
def get_flights():
    return {
        "flights": [
            {"number": "AA101", "route": "JFK to LAX", "status": "On Time", "gate": "B12"},
            {"number": "AA202", "route": "ORD to DFW", "status": "Delayed 30min", "gate": "C8"},
            {"number": "AA303", "route": "MIA to BOS", "status": "Cancelled", "gate": "N/A"},
            {"number": "AA404", "route": "SEA to DEN", "status": "On Time", "gate": "A5"},
            {"number": "AA505", "route": "SFO to JFK", "status": "On Time", "gate": "D3"},
            {"number": "AA606", "route": "LAX to ORD", "status": "Delayed 15min", "gate": "E7"},
        ],
        "last_updated": datetime.now().isoformat()
    }

@app.get("/tools")
def get_tools():
    return {
        "available_tools": [
            {"name": t.name, "description": t.description}
            for t in tools
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
