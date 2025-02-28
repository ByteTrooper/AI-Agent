import streamlit as st
import ollama
import json
import os
import re
import datetime
import random
from typing import List, Dict, Any, Optional, Tuple

# Configuration
MODEL_NAME = "llama3.1:8b"
DATABASE_FILE = "restaurants_db.json"

# States for conversation flow
class State:
    GREETING = "Greeting"
    INTENT_DETECTION = "IntentDetection"
    FIND_RESTAURANT = "FindRestaurant"
    RESTAURANT_SUGGESTION = "RestaurantSuggestion"
    RESTAURANT_DETAILS = "RestaurantDetails"
    MAKE_RESERVATION = "MakeReservation"
    DATA_COLLECTION = "DataCollection"
    NAME_PROMPT = "NamePrompt"
    NAME_RECEIVED = "NameReceived"
    DATETIME_PROMPT = "DateTimePrompt"
    DATETIME_RECEIVED = "DateTimeReceived"
    PARTY_PROMPT = "PartyPrompt"
    PARTY_RECEIVED = "PartyReceived"
    CONFIRM_RESERVATION = "ConfirmReservation"
    RESERVATION_CONFIRMED = "ReservationConfirmed"
    DATABASE_PUSH = "DatabasePush"
    RESERVATION_SUCCESS = "ReservationSuccess"
    ERROR_HANDLING = "ErrorHandling"
    RESERVATION_RETRY = "ReservationRetry"
    SUPPORT = "Support"
    THANK_YOU = "ThankYou"
    NORMAL_CONVERSATION = "NormalConversation"

# Create restaurants database if it doesn't exist
def create_database():
    if os.path.exists(DATABASE_FILE):
        return
    
    # Bengaluru neighborhoods
    areas = [
        "Indiranagar", "Koramangala", "MG Road", "Church Street", "HSR Layout",
        "Whitefield", "Jayanagar", "JP Nagar", "Malleshwaram", "Lavelle Road",
        "Brigade Road", "UB City", "Bannerghatta Road", "Electronic City", "Yelahanka"
    ]
    
    # Cuisine types
    cuisines = [
        "South Indian", "North Indian", "Chinese", "Italian", "Continental",
        "Japanese", "Mexican", "Mediterranean", "Thai", "Fusion",
        "Bengali", "Chettinad", "Punjabi", "Mughlai", "Goan"
    ]
    
    # Seating arrangements
    seating_options = [
        "Indoor", "Outdoor", "Rooftop", "Private cabins", "Bar seating",
        "Community tables", "Window seating", "Booth seating", "Terrace", "Garden"
    ]
    
    # Price ranges (for 2 people in INR)
    price_ranges = [
        "₹500-1000", "₹1000-1500", "₹1500-2000", "₹2000-2500", "₹2500-3000",
        "₹3000-4000", "₹4000-5000", "₹5000-6000", "₹6000-8000", "₹8000+"
    ]
    
    # Restaurant names
    names = [
        "Spice Garden", "Bengaluru Bytes", "Silicon Spices", "Garden City Grill", 
        "Cubbon Cuisine", "Lalbagh Lunches", "Namma Kitchen", "Tech Park Tavern",
        "Kodava Kitchen", "Brigade Bistro", "Infosys Eatery", "Startup Sizzlers",
        "Royal Repast", "Majestic Meals", "Palace Platters", "Coffee Connect",
        "Monsoon Masala", "Mango Mantra", "Windmill Wok", "Bamboo Bytes"
    ]
    
    # Generate 20 restaurants with varied attributes
    restaurants = []
    for i in range(20):
        restaurant = {
            "id": i + 1,
            "name": names[i],
            "cuisine": random.choice(cuisines),
            "location": random.choice(areas),
            "price_range": random.choice(price_ranges),
            "rating": round(random.uniform(3.5, 4.9), 1),
            "seating_arrangements": random.sample(seating_options, random.randint(2, 5)),
            "capacity": random.randint(20, 200),
            "opening_hours": {
                "weekdays": f"{random.randint(8, 12)}:00 AM - {random.randint(9, 11)}:00 PM",
                "weekends": f"{random.randint(8, 12)}:00 AM - {random.randint(9, 11)}:00 PM"
            },
            "specialties": [f"Signature dish {i+1}", f"Special drink {i+1}"],
            "address": f"{random.randint(1, 100)}, {random.choice(['Main', 'Cross', 'Avenue', 'Street'])}, {random.choice(areas)}, Bengaluru",
            "contact": f"+91 {random.randint(6000000000, 9999999999)}",
            "reservations": []
        }
        restaurants.append(restaurant)
    
    # Save to JSON file
    with open(DATABASE_FILE, "w") as f:
        json.dump(restaurants, f, indent=4)

# Load restaurants database
def load_database():
    with open(DATABASE_FILE, "r") as f:
        return json.load(f)

# Save restaurants database
def save_database(restaurants):
    with open(DATABASE_FILE, "w") as f:
        json.dump(restaurants, f, indent=4)

# Get response from Llama 3.1 model with database metadata
def get_llm_response(prompt: str, system_prompt: str = None, restaurants_data: dict = None) -> str:
    # Create a data-aware system prompt with accurate restaurant count
    if restaurants_data:
        data_metadata = f"""
        IMPORTANT: You have access to exactly {len(restaurants_data)} restaurants in the database.
        Available cuisines: {', '.join(set(r['cuisine'] for r in restaurants_data))}
        Available locations: {', '.join(set(r['location'] for r in restaurants_data))}
        Available price ranges: {', '.join(set(r['price_range'] for r in restaurants_data))}
        """
        
        if system_prompt:
            system_prompt = system_prompt + "\n" + data_metadata
        else:
            system_prompt = data_metadata
    
    if system_prompt:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
    else:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    
    return response['message']['content']

# Intent detection function
def detect_intent(user_input: str) -> str:
    system_prompt = """
    You are an AI assistant for a restaurant booking system. Analyze the user input and determine their intent.
    Respond with exactly one of these intents: 'restaurant_search', 'reservation', 'details', 'normal_conversation'.
    Don't explain your reasoning, just respond with the intent.
    """
    
    prompt = f"Determine the intent from this user input: '{user_input}'"
    response = get_llm_response(prompt, system_prompt).strip().lower()
    
    if 'restaurant_search' in response or 'search' in response:
        return 'restaurant_search'
    elif 'reservation' in response or 'book' in response:
        return 'reservation'
    elif 'details' in response or 'information' in response:
        return 'details'
    else:
        return 'normal_conversation'

# Extract restaurant search parameters
def extract_search_params(user_input: str) -> Dict[str, Any]:
    system_prompt = """
    You are an AI assistant for a restaurant booking system. Extract search parameters from the user input.
    Output ONLY a JSON object with these fields:
    {
        "cuisine": "extracted cuisine or null",
        "location": "extracted location or null",
        "price_range": "extracted price range or null",
        "seating": "extracted seating preference or null"
    }
    Don't include any other text in your response. Only return valid JSON.
    """
    
    prompt = f"Extract search parameters from: '{user_input}'"
    response = get_llm_response(prompt, system_prompt)
    
    try:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return {}
    except:
        return {}

# Find restaurants based on search parameters
def find_restaurants(params: Dict[str, Any], restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = restaurants.copy()
    
    if params.get('cuisine') and params['cuisine'] != 'null':
        filtered = [r for r in filtered if params['cuisine'].lower() in r['cuisine'].lower()]
    
    # Improved location matching in find_restaurants function
    if params.get('location') and params['location'] != 'null':
        location_query = params['location'].lower()
        # More flexible location matching
        filtered = [r for r in filtered if location_query in r['location'].lower() or 
                    location_query in r['address'].lower()]
    
    if params.get('price_range') and params['price_range'] != 'null':
        # Simple matching for now
        filtered = [r for r in filtered if params['price_range'].lower() in r['price_range'].lower()]
    
    if params.get('seating') and params['seating'] != 'null':
        filtered = [r for r in filtered if any(params['seating'].lower() in s.lower() for s in r['seating_arrangements'])]
    
    # Return top restaurants by rating if we have too many
    if len(filtered) > 5:
        filtered.sort(key=lambda x: x['rating'], reverse=True)
        return filtered[:5]
    
    # If no results, return some recommended ones
    if not filtered:
        return sorted(restaurants, key=lambda x: x['rating'], reverse=True)[:3]
    
    return filtered

# Generate restaurant suggestions text
def generate_restaurant_suggestions(restaurants: List[Dict[str, Any]], all_restaurants: List[Dict[str, Any]]) -> str:
    system_prompt = f"""
    You are Alfred, an AI assistant for a restaurant booking system in Bengaluru. Create an enticing and informative suggestion
    message for these restaurants. Keep it friendly and conversational, but focused.
    
    IMPORTANT: You have access to exactly {len(all_restaurants)} restaurants in the database. DO NOT claim or imply
    that you have more restaurants than this number.
    """
    
    prompt = f"""Based on the user's preferences, here are some restaurant options:
    {json.dumps(restaurants, indent=2)}
    
    Generate a helpful response suggesting these restaurants. Include their name, cuisine, location, and one unique feature for each.
    Keep it concise (max 200 words).
    """
    
    return get_llm_response(prompt, system_prompt, all_restaurants)

# Generate restaurant details
def generate_restaurant_details(restaurant: Dict[str, Any], all_restaurants: List[Dict[str, Any]]) -> str:
    system_prompt = f"""
    You are Alfred, an AI assistant for a restaurant booking system in Bengaluru. Create a detailed and enticing description
    of this restaurant. Focus on what makes it special, its atmosphere, food options, and practical details.
    
    IMPORTANT: You have access to exactly {len(all_restaurants)} restaurants in the database. DO NOT claim or imply
    that you have more restaurants than this number.
    """
    
    prompt = f"""Please provide detailed information about this restaurant:
    {json.dumps(restaurant, indent=2)}
    
    Generate a compelling description highlighting its key features, cuisine, location, seating arrangements, and any specialties.
    Keep it informative and attractive but factual. (max 200 words)
    """
    
    return get_llm_response(prompt, system_prompt, all_restaurants)

# Modified parse_date_time function with correct weekday calculation
def parse_date_time(user_input: str) -> Tuple[Optional[datetime.datetime], str]:
    system_prompt = """
    You are an AI assistant for a restaurant booking system. Extract date and time information from user input.
    Output ONLY a JSON object with these fields:
    {
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "confidence": "high/medium/low"
    }
    If you can't determine the date or time, use null for that field.
    Don't include any other text in your response. Only return valid JSON.
    """
    
    prompt = f"Extract date and time from: '{user_input}'"
    response = get_llm_response(prompt, system_prompt)
    
    try:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            
            if parsed.get('date') and parsed.get('time') and parsed['date'] != "null" and parsed['time'] != "null":
                dt_string = f"{parsed['date']} {parsed['time']}"
                try:
                    dt = datetime.datetime.strptime(dt_string, "%Y-%m-%d %H:%M")
                    
                    # Ensure the weekday name is correctly calculated
                    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    weekday_num = dt.weekday()  # 0 = Monday, 1 = Tuesday, etc.
                    
                    # Double-check that our internal calculations are correct
                    # Do not trust the LLM's calculation of weekday
                    if dt.strftime("%A") != weekday_names[weekday_num]:
                        # If Python's built-in weekday name doesn't match our calculation,
                        # force the correct weekday name based on the date
                        dt_formatted = dt.strftime("%B %d at %I:%M %p")
                        return dt, f"Got it! Reservation for {weekday_names[weekday_num]}, {dt_formatted}."
                    
                    return dt, ""
                except ValueError:
                    return None, "I couldn't understand that date format. Please specify a date and time."
            else:
                return None, "I couldn't determine the date or time. Please specify both."
        else:
            return None, "I couldn't parse that. Please specify when you'd like to make your reservation."
    except:
        return None, "I'm having trouble understanding the date and time. Please use a format like 'tomorrow at 7 PM'."
    
# Parse party size from user input
def parse_party_size(user_input: str) -> Tuple[Optional[int], str]:
    system_prompt = """
    You are an AI assistant for a restaurant booking system. Extract the party size (number of people) from user input.
    Output ONLY a JSON object with these fields:
    {
        "party_size": number or null,
        "confidence": "high/medium/low"
    }
    Don't include any other text in your response. Only return valid JSON.
    """
    
    prompt = f"Extract party size from: '{user_input}'"
    response = get_llm_response(prompt, system_prompt)
    
    try:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            
            if parsed.get('party_size') and parsed['party_size'] != "null":
                return int(parsed['party_size']), ""
            else:
                return None, "I couldn't determine the party size. How many people will be dining?"
        else:
            return None, "I couldn't parse that. Please specify how many people are in your party."
    except:
        return None, "I'm having trouble understanding the party size. Please tell me how many people will be dining."

# Add reservation to database
def add_reservation(restaurants, restaurant_id, name, datetime_obj, party_size):
    for restaurant in restaurants:
        if restaurant['id'] == restaurant_id:
            reservation_id = len(restaurant['reservations']) + 1
            reservation = {
                "id": reservation_id,
                "name": name,
                "datetime": datetime_obj.strftime("%Y-%m-%d %H:%M"),
                "party_size": party_size,
                "status": "confirmed"
            }
            restaurant['reservations'].append(reservation)
            save_database(restaurants)
            return True, reservation
    return False, None

# Generate a normal conversation response
def generate_conversation_response(user_input: str, chat_history: List[Dict[str, str]], restaurants: List[Dict[str, Any]]) -> str:
    system_prompt = f"""
    You are Alfred, a friendly AI assistant for a restaurant booking system in Bengaluru.
    Respond to the user in a helpful, concise, and conversational way. If they're asking about restaurants 
    or reservations, guide them toward using those features of the system.
    
    IMPORTANT: You have access to exactly {len(restaurants)} restaurants in the database. DO NOT claim or imply
    that you have more restaurants than this number. Only reference the actual number of available restaurants.
    
    Keep your response under 100 words unless you need to provide detailed information.
    """
    
    # Create a formatted chat history for context
    history_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history[-3:]])
    
    prompt = f"""Recent conversation:
    {history_text}
    
    User: {user_input}
    
    Please provide a helpful response as the restaurant booking assistant:
    """
    
    return get_llm_response(prompt, system_prompt, restaurants)

# Enhanced restaurant matching using LLM
def match_restaurant_from_input(user_input, available_restaurants, all_restaurants):
    system_prompt = f"""
    You are an AI assistant for a restaurant booking system. The user is referring to a restaurant in their message.
    From the list of available restaurants, determine which one they're most likely referring to.
    
    IMPORTANT: You have access to exactly {len(all_restaurants)} restaurants in the database. Only match restaurants
    from the provided list of available restaurants.
    
    Return ONLY a JSON with the restaurant ID:
    {{"restaurant_id": number or null}}
    If you can't confidently match a restaurant, return null.
    """
    
    prompt = f"""
    User message: '{user_input}'
    Available restaurants: {json.dumps([{'id': r['id'], 'name': r['name']} for r in available_restaurants])}
    
    Which restaurant is the user referring to?
    """
    
    response = get_llm_response(prompt, system_prompt, all_restaurants)
    
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            return parsed.get('restaurant_id')
    except:
        pass
    
    return None

# Enhanced intent detection with context
def detect_intent_with_context(user_input, context=None, restaurants=None):
    system_prompt = f"""
    You are an AI assistant for a restaurant booking system. Analyze the user input and determine their intent.
    Consider any context provided. 
    
    IMPORTANT: You have access to exactly {len(restaurants)} restaurants in the database. DO NOT claim or imply
    that you have more restaurants than this number.
    
    Return ONLY a JSON object with:
    {{
        "intent": "restaurant_search" or "reservation" or "details" or "normal_conversation",
        "confidence": "high" or "medium" or "low",
        "restaurant_id": number or null (if they're referring to a specific restaurant)
    }}
    """
    
    context_str = f"Context: {context}\n\n" if context else ""
    prompt = f"{context_str}Determine the intent from this user input: '{user_input}'"
    
    response = get_llm_response(prompt, system_prompt, restaurants)
    
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except:
        # Fallback
        return {"intent": "normal_conversation", "confidence": "low", "restaurant_id": None}


# Modified main function to prevent duplicate processing
def main():
    # Create database if it doesn't exist
    create_database()
    
    # Page configuration
    st.set_page_config(
        page_title="Alfred - Bengaluru Restaurant Assistant",
        page_icon="🍽️",
        layout="centered"
    )
    
    # Initialize session state variables
    if 'state' not in st.session_state:
        st.session_state.state = State.GREETING
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_restaurant_id' not in st.session_state:
        st.session_state.current_restaurant_id = None
    if 'reservation_info' not in st.session_state:
        st.session_state.reservation_info = {
            "name": None,
            "datetime": None,
            "party_size": None
        }
    if 'filtered_restaurants' not in st.session_state:
        st.session_state.filtered_restaurants = []
    if 'last_processed_input' not in st.session_state:
        st.session_state.last_processed_input = None
    
    # Load database
    restaurants = load_database()
    
    # App title
    st.title("Alfred 🍽️")
    st.subheader("Your Bengaluru Restaurant Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.write(message["content"])
    
    # Initial greeting
    if st.session_state.state == State.GREETING and not st.session_state.chat_history:
        greeting = "Hello! I'm Alfred, your Bengaluru restaurant assistant. I can help you find restaurants and make reservations. What are you looking for today?"
        st.session_state.chat_history.append({"role": "assistant", "content": greeting})
        st.session_state.state = State.INTENT_DETECTION
        with st.chat_message("assistant"):
            st.write(greeting)
    
    # Get user input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Check if we've already processed this exact input to prevent duplicates
        if user_input != st.session_state.last_processed_input:
            st.session_state.last_processed_input = user_input
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process based on current state
            response = process_state(user_input, restaurants)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Modified process_state function to prevent duplicate state processing
def process_state(user_input, restaurants):
    current_state = st.session_state.state
    next_state = None
    response = None
    
    # Intent detection state
    if current_state == State.INTENT_DETECTION:
        intent = detect_intent(user_input)
        
        if intent == 'restaurant_search':
            next_state = State.FIND_RESTAURANT
            # Don't process again immediately
        
        elif intent == 'reservation':
            if st.session_state.current_restaurant_id:
                next_state = State.MAKE_RESERVATION
                response = "Great! Let's make a reservation. Under what name should I make the reservation?"
            else:
                next_state = State.FIND_RESTAURANT
                response = "I'd be happy to help you make a reservation. First, let's find a restaurant. Do you have any preferences for cuisine, location, or price range?"
        
        elif intent == 'details':
            if st.session_state.current_restaurant_id:
                next_state = State.RESTAURANT_DETAILS
                # Don't process again immediately
            else:
                next_state = State.FIND_RESTAURANT
                response = "I'd be happy to provide details about a restaurant. Which restaurant are you interested in?"
        
        else:  # normal_conversation
            next_state = State.NORMAL_CONVERSATION
            response = generate_conversation_response(user_input, [
                {"user": msg["content"], "assistant": st.session_state.chat_history[i+1]["content"]} 
                for i, msg in enumerate(st.session_state.chat_history[:-1:2])
            ], restaurants)
    
    # Find restaurant state
    elif current_state == State.FIND_RESTAURANT:
        search_params = extract_search_params(user_input)
        filtered = find_restaurants(search_params, restaurants)
        
        st.session_state.filtered_restaurants = filtered
        next_state = State.RESTAURANT_SUGGESTION
        
        response = generate_restaurant_suggestions(filtered, restaurants)
    
    # Restaurant suggestion state
    elif current_state == State.RESTAURANT_SUGGESTION:
        # First try a simple direct name match
        restaurant_id = None
        for r in st.session_state.filtered_restaurants:
            if r["name"].lower() in user_input.lower():
                restaurant_id = r["id"]
                break
        
        # If direct match doesn't work, use the context-based matching
        if not restaurant_id:
            context = {
                "recent_suggestions": [r["name"] for r in st.session_state.filtered_restaurants],
                "last_message": st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) > 1 else ""
            }
            
            # Let the LLM detect intent with context
            intent_data = detect_intent_with_context(user_input, json.dumps(context), restaurants)
            restaurant_id = intent_data.get("restaurant_id")
        
        # If we found a restaurant match
        if restaurant_id:
            st.session_state.current_restaurant_id = restaurant_id
            restaurant = next((r for r in restaurants if r["id"] == st.session_state.current_restaurant_id), None)
            
            if "reserv" in user_input.lower() or "book" in user_input.lower():
                next_state = State.MAKE_RESERVATION
                response = f"Great! I'll make a reservation at {restaurant['name']} for you. Under what name should I make the reservation?"
            else:
                next_state = State.RESTAURANT_DETAILS
                response = generate_restaurant_details(restaurant, restaurants)
        
        # If no match but reservation intent
        elif "reserv" in user_input.lower() or "book" in user_input.lower():
            response = "Which of these restaurants would you like to make a reservation for? Please specify the restaurant name clearly."
            # Stay in the same state
        
        # Otherwise continue with normal flow
        else:
            next_state = State.INTENT_DETECTION
            # Will process in the next cycle
    
    # Restaurant details state
    elif current_state == State.RESTAURANT_DETAILS:
        # Check if user wants to make a reservation
        intent = detect_intent(user_input)
        
        if 'reserv' in user_input.lower() or 'book' in user_input.lower() or intent == 'reservation':
            next_state = State.MAKE_RESERVATION
            response = "Great! I'd be happy to make a reservation for you. Under what name should I make the reservation?"
        else:
            # They're satisfied with the information or have another question
            next_state = State.INTENT_DETECTION
            # Will process in the next cycle
    
    # Make reservation state (start data collection)
    elif current_state == State.MAKE_RESERVATION:
        next_state = State.NAME_PROMPT
        response = "Under what name should I make the reservation?"
    
    # Name prompt state
    elif current_state == State.NAME_PROMPT:
        st.session_state.reservation_info["name"] = user_input
        next_state = State.DATETIME_PROMPT  # Skip the intermediate NAME_RECEIVED state
        response = f"Thank you, {user_input}. When would you like to make your reservation? Please specify the date and time."
    
    # Datetime prompt state
    elif current_state == State.DATETIME_PROMPT:
        datetime_obj, error_msg = parse_date_time(user_input)
        
        if datetime_obj:
            st.session_state.reservation_info["datetime"] = datetime_obj
            next_state = State.PARTY_PROMPT  # Skip the intermediate DATETIME_RECEIVED state
            response = f"Got it! Reservation for {datetime_obj.strftime('%A, %B %d at %I:%M %p')}. How many people will be in your party?"
        else:
            response = error_msg
            # Stay in the same state
    
    # Party prompt state
    elif current_state == State.PARTY_PROMPT:
        party_size, error_msg = parse_party_size(user_input)
        
        if party_size:
            st.session_state.reservation_info["party_size"] = party_size
            next_state = State.CONFIRM_RESERVATION  # Skip the intermediate PARTY_RECEIVED state
            
            # Get restaurant name
            restaurant = next((r for r in restaurants if r["id"] == st.session_state.current_restaurant_id), {"name": "the restaurant"})
            
            response = f"Great! Let me confirm your reservation:\n\nName: {st.session_state.reservation_info['name']}\nRestaurant: {restaurant['name']}\nDate & Time: {st.session_state.reservation_info['datetime'].strftime('%A, %B %d at %I:%M %p')}\nParty Size: {party_size} people\n\nIs this information correct? (yes/no)"
        else:
            response = error_msg
            # Stay in the same state
    
    # Confirm reservation state
    elif current_state == State.CONFIRM_RESERVATION:
        if 'yes' in user_input.lower() or 'correct' in user_input.lower() or 'confirm' in user_input.lower():
            next_state = State.RESERVATION_CONFIRMED
            response = "Thank you for confirming. I'm processing your reservation now..."
        else:
            next_state = State.MAKE_RESERVATION
            response = "Let's try again. Under what name should I make the reservation?"
    
    # Reservation confirmed state (transition to database push)
    elif current_state == State.RESERVATION_CONFIRMED:
        next_state = State.DATABASE_PUSH
        
        success, reservation = add_reservation(
            restaurants,
            st.session_state.current_restaurant_id,
            st.session_state.reservation_info["name"],
            st.session_state.reservation_info["datetime"],
            st.session_state.reservation_info["party_size"]
        )
        
        if success:
            next_state = State.RESERVATION_SUCCESS
            
            # Get restaurant name
            restaurant = next((r for r in restaurants if r["id"] == st.session_state.current_restaurant_id), {"name": "the restaurant"})
            
            response = f"Your reservation at {restaurant['name']} has been confirmed! We look forward to serving you on {st.session_state.reservation_info['datetime'].strftime('%A, %B %d at %I:%M %p')}. Is there anything else you would like to know?"
        else:
            next_state = State.ERROR_HANDLING
            response = "I'm sorry, there was an error processing your reservation. Would you like to try again or speak with customer support?"
    
    # Error handling state
    elif current_state == State.ERROR_HANDLING:
        if 'try' in user_input.lower() or 'again' in user_input.lower():
            next_state = State.RESERVATION_RETRY
            response = "Let's try making your reservation again. Under what name should I make the reservation?"
        else:
            next_state = State.SUPPORT
            response = "I'll connect you with customer support. Please call +91 80 12345678 during business hours (9 AM - 6 PM) for assistance with your reservation."
    
    # Reservation retry state
    elif current_state == State.RESERVATION_RETRY:
        next_state = State.MAKE_RESERVATION
        response = "Let's start over with your reservation. Under what name should I make the reservation?"
    
    # Support state
    elif current_state == State.SUPPORT:
        next_state = State.INTENT_DETECTION
        response = "Our customer support team is available to help you. Is there anything else I can assist you with?"
    
    # Reservation success state
    elif current_state == State.RESERVATION_SUCCESS:
        next_state = State.THANK_YOU
        response = "Thank you for using Alfred! Your reservation has been confirmed. Is there anything else you would like assistance with?"
    
    # Thank you state
    elif current_state == State.THANK_YOU:
        next_state = State.INTENT_DETECTION
        # Will process in the next cycle
    
    # Normal conversation state
    elif current_state == State.NORMAL_CONVERSATION:
        next_state = State.INTENT_DETECTION
        # Will process in the next cycle
    
    # Update the state if a new state was set
    if next_state:
        st.session_state.state = next_state
    
    # If we don't have a response yet and we've changed state, process the new state
    if response is None and next_state and next_state != current_state:
        return process_state(user_input, restaurants)
    
    # Return the response
    return response or "I'm not sure what you're looking for. Would you like to search for restaurants or make a reservation?"

if __name__ == "__main__":
    main()