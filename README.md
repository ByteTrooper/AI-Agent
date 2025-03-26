# AI-Agent [Restaurant Chatbot]

This project is a restaurant chatbot that helps users search for restaurants and make reservations. It uses state management to handle different conversation states and process user inputs.

🎥 [Watch the Demo](https://github.com/ByteTrooper/AI-Agent/raw/main/working%20demo%201.mp4)

📑 **Notion Document:** [Click here](https://www.notion.so/AI-Agent-Challenge-Restaurant-Reservation-System-Business-Strategy-Document-1a8ccb77163580b59f86ed667ecfc3d2?pvs=4)




![Screenshot 2025-02-27 014725](https://github.com/user-attachments/assets/f1d44bc6-cfbd-4f65-9777-862af4e195ff)


<img src="https://github.com/user-attachments/assets/21612832-906d-4273-ac6b-3de54275fc13" width="400" height="auto">



## How the Agent Works
The restaurant chatbot agent is designed to handle user interactions for searching restaurants and making reservations. It uses a state management system to process user inputs and transition between different conversation states. Here's a detailed description of how the agent works:

1)State Management:

The chatbot maintains different states to handle various parts of the conversation.
States include THANK_YOU, NORMAL_CONVERSATION, INTENT_DETECTION, and others.
The current state determines how the chatbot processes the user's input.

2)State Transitions:

When the chatbot is in the THANK_YOU state, it transitions to the INTENT_DETECTION state for further processing in the next cycle.
Similarly, in the NORMAL_CONVERSATION state, it transitions to the INTENT_DETECTION state.
The state is updated in the session state to keep track of the conversation flow.

3)Processing User Input:

If the chatbot has not generated a response yet and the state has changed, it processes the new state by calling the process_state function with the user's input and restaurant data.
The process_state function handles the logic for each state and generates an appropriate response.

4)Generating Responses:

If a response is generated, it is returned to the user.
If no response is generated and the state has not changed, the chatbot prompts the user with a default message asking if they want to search for restaurants or make a reservation.

5)Main Function:

The main function is the entry point of the chatbot application.
It initializes the chatbot and starts the conversation loop.

## Features

- Search for restaurants
- Make reservations
- Handle different conversation states

## Installation

1. Clone the repository:
    ```sh
    git clone [https://github.com/ByteTrooper/AI-Agent]
    cd AI-Agent
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the chatbot:
```sh
python main.py
