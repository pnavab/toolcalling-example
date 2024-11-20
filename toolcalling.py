import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

def get_weather(location):
    """Look up weather for a given location using OpenWeatherMap API"""
    try:
        # First, get coordinates for the location
        geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geocoding_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return f"Could not find location: {location}"
            
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Get weather data using coordinates
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        
        return f"Current weather in {location}: {description}, temperature is {temp}°C with {humidity}% humidity"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# Create LangChain tools
tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Useful for getting weather information for a location. Input should be a city name."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def process_query(query: str) -> str:
    """Process a query using the LangChain agent"""
    try:
        response = agent.invoke(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Example usage
if __name__ == "__main__":
    print("\nTesting LangChain agent:")
    test_query = "What's the purpose of water bottles?"
    print(f"Query: {test_query}")
    response = process_query(test_query)['output']
    print(f"Response: {response}") 
