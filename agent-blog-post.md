# Building AI Agents from Scratch: No Frameworks Needed

In the rapidly evolving landscape of AI development, frameworks like LangChain and LlamaIndex have made it easier to build complex AI agents. However, these frameworks can sometimes obscure the core principles of how agents work and limit flexibility. In this post, we'll explore how to build a functional AI agent from scratch, providing a clearer understanding of the underlying architecture while maintaining full control over your implementation.

We'll use a restaurant recommendation agent as our practical example, referencing the implementation from [ByteTrooper's AI-Agent](https://github.com/ByteTrooper/AI-Agent).

## What Are AI Agents?

AI agents are systems that can observe their environment, make decisions, and take actions to achieve specific goals. Unlike traditional AI systems that perform specific, isolated tasks, agents can interact with their environment and adapt their behavior based on feedback and changing conditions.

An AI agent typically consists of:
1. A foundation model (like an LLM) that serves as the "brain"
2. Tools and APIs the agent can use to interact with the world
3. Memory systems to retain information across interactions
4. Planning capabilities to break down complex tasks
5. Learning mechanisms to improve performance over time

AI agents differ from traditional chatbots by having more autonomy, tool-using capabilities, and the ability to plan multi-step processes to accomplish complex tasks.

### The Evolution from Chatbots to Agents

To understand AI agents better, it's helpful to see how they evolved from earlier conversational AI:

- **Rule-based Chatbots**: Used predefined rules and pattern matching to respond to users
- **ML-powered Chatbots**: Used machine learning to improve natural language understanding
- **LLM-based Assistants**: Used large language models to generate more natural responses
- **AI Agents**: Extended LLM capabilities with tool use, reasoning, planning, and feedback loops

Agents represent a significant leap forward by combining the natural language capabilities of LLMs with the ability to execute actions in the world through tools and APIs.

### Why Agents Matter: The Tool-Using Paradigm

The true power of AI agents comes from their ability to bridge the gap between language understanding and practical action. Consider these fundamental capabilities:

1. **Tool Use**: Agents can call external tools, APIs, and functions to perform actions beyond text generation
2. **Planning**: They can break complex tasks into steps and execute them sequentially
3. **Feedback Loops**: They can evaluate the results of their actions and adjust accordingly
4. **Specialized Expertise**: They can be optimized for specific domains (like restaurant recommendations)
5. **Contextual Memory**: They can maintain information across multiple turns of conversation

These capabilities enable agents to solve problems that would be impossible for traditional systems.

### Real-World Applications of Agents

AI agents are already transforming numerous industries:

- **Customer Service**: Agents that can not only answer questions but also process returns, check inventory, and place orders
- **Personal Assistants**: Calendar management, email summarization, and task prioritization
- **Healthcare**: Appointment scheduling, medication reminders, and preliminary symptom analysis
- **E-commerce**: Product recommendation, comparison shopping, and transaction processing
- **Education**: Personalized tutoring with adaptive learning paths

Our restaurant agent example illustrates just one specialized use case, but the same architectural principles apply across domains.

### The Agent Loop: Perception, Reasoning, Action, Learning

Drawing from the Hugging Face Agents Course, an agent operates in a continuous loop:

1. **Perception**: The agent receives observations from the environment (text, images, structured data)
2. **Reasoning**: It updates its internal state based on observations and decides on the next action
3. **Action**: It executes operations through tools, APIs, or generates responses
4. **Learning**: It observes the results and adapts its behavior accordingly

This cycle repeats continuously, with each interaction potentially involving multiple reasoning and action steps.

Let's visualize this with our restaurant agent:
- **Perception**: User says "I need a table for 4 tomorrow evening"
- **Reasoning**: Agent identifies intent as "make reservation" and determines required information
- **Action**: Agent calls the reservation tool, which might then ask for missing details like time and name
- **Learning**: The agent adapts based on user corrections (e.g., if user changes the party size)

### Types of Agents

The Hugging Face course highlights different categories of agents:

- **Reactive Agents**: Respond directly to the current state without considering history
  - Example: A simple restaurant FAQ bot that answers questions based on predefined patterns

- **Planning Agents**: Create sequences of actions to achieve goals
  - Example: Our restaurant agent that breaks down the reservation process into steps

- **Learning Agents**: Improve performance over time through experience
  - Example: A recommendation system that refines suggestions based on customer feedback

- **Multi-Agent Systems**: Multiple agents that collaborate or compete to solve problems
  - Example: A restaurant system with separate reservation, menu, and payment agents


## Core Components of AI Agents

Based on both practical experience and the Hugging Face Agents Course, a comprehensive AI agent architecture includes:

1. **Foundation Model**: The "brain" that processes input and generates responses (e.g., Llama 3.1-8B in our example)
2. **Memory System**: Mechanisms to maintain conversation history and relevant information
3. **Tools**: Specific functions the agent can invoke to perform actions
4. **Controller/Orchestrator**: Coordinates the flow between components
5. **User Interface**: Facilitates interaction with users
6. **Learning Mechanism**: Allows the agent to improve over time

### The Central Role of the Language Model

The language model serves as the cognitive engine of an AI agent. It performs several critical functions:

- **Intent Recognition**: Understanding what the user wants to accomplish
- **Reasoning**: Working through problems in a step-by-step manner
- **Decision Making**: Choosing appropriate actions based on context
- **Natural Language Generation**: Producing human-like responses

The quality of the language model significantly impacts agent performance. Modern models like Llama 3.1, GPT-4, and Claude have drastically improved agents' capabilities through their enhanced reasoning and instruction-following abilities.

### Memory: Beyond Simple Context Windows

Agent memory can be implemented in several ways:

- **Short-term Memory**: Recent conversation history (typically stored in context)
- **Working Memory**: Current task state and intermediate reasoning steps
- **Long-term Memory**: Persistent information about users and past interactions

Advanced agent architectures often implement vector databases for semantic search across past interactions, allowing agents to reference relevant information from much earlier in the conversation history.

### Tools: Extending the Agent's Capabilities

Tools transform agents from conversational systems into action-oriented assistants. A well-designed tool should:

- Have a clear, single-purpose function
- Accept structured inputs and return structured outputs
- Handle errors gracefully
- Be composable with other tools

In production environments, tools might include:
- Database queries
- API calls to external services
- Data processing functions
- File I/O operations
- Authentication systems

### The Controller: The Decision-Making Hub

The controller is perhaps the most underappreciated component of agent architecture. It must:

1. Determine when to use tools vs. direct LLM responses
2. Manage the sequence of tool calls
3. Handle errors and edge cases
4. Maintain state across multiple turns
5. Optimize for user experience and system performance

Different controller architectures offer different benefits:
- **Rule-based controllers**: Simple but inflexible
- **LLM-powered controllers**: Flexible but potentially inconsistent
- **Hybrid approaches**: Combining structured rules with LLM flexibility

## Building Our Restaurant Agent

Let's examine how each component is implemented in our example restaurant agent:

### 1. Setting Up the Model

```python
# Using Llama 3.1-8B locally
# In main.py, we initialize the model
```

When building from scratch, you have the flexibility to choose any model and integration method:
- Local models (Llama, Mistral, etc.)
- API-based models (OpenAI, Anthropic, etc.)
- Custom fine-tuned models

The key is setting up a consistent interface for sending prompts and receiving responses.

<img src="https://github.com/user-attachments/assets/3a778e14-ce9f-43f5-8bd8-66101cb77358" alt="image" width="300">

### 2. Implementing Context Management

Maintaining conversation history is crucial for coherent interactions:

```python
# Simple context implementation
class ConversationContext:
    def __init__(self, max_turns=10):
        self.history = []
        self.max_turns = max_turns
    
    def add_exchange(self, user_input, agent_response):
        self.history.append({"user": user_input, "agent": agent_response})
        # Trim if exceeding max context
        if len(self.history) > self.max_turns:
            self.history = self.history[1:]
    
    def get_formatted_context(self):
        formatted = ""
        for exchange in self.history:
            formatted += f"User: {exchange['user']}\nAgent: {exchange['agent']}\n\n"
        return formatted
```

This simple context manager tracks conversation turns and formats them for inclusion in prompts.

### 3. Defining Tools

Tools are functions that extend the agent's capabilities beyond just conversation:

```python
def find_intent(user_input, context):
    # Prompt the LLM to identify user intent
    prompt = f"""
    Based on the following user input and conversation context, 
    identify the primary intent (RECOMMEND_RESTAURANT, MAKE_RESERVATION, or GENERAL_INQUIRY):
    
    Context: {context}
    User input: {user_input}
    
    Intent:
    """
    return llm(prompt).strip()

def suggest_restaurant(preferences, context):
    # Tool to recommend restaurants based on user preferences
    prompt = f"""
    Based on these user preferences: {preferences}
    Suggest 3 restaurants with details about cuisine, price range, and seating options.
    """
    return llm(prompt).strip()

def make_reservation(user_input, context):
    # Multi-turn reservation tool
    # Extract information or ask for missing details
    required_info = ["name", "date", "time", "party_size"]
    # Logic to extract or request missing information
    # ...
```

Unlike framework-based implementations, our custom tools:
- Have full flexibility in implementation
- Can be synchronous or asynchronous
- Are tailored exactly to our application's needs
- Can include complex logic like multi-turn interactions

<img src="[https://github.com/user-attachments/assets/3a778e14-ce9f-43f5-8bd8-66101cb77358](https://github.com/user-attachments/assets/e98e6f11-97b5-45b5-bd11-69c9a8e4f7df)" alt="image" width="300">

### 4. Building the Controller

The controller is the orchestration layer that determines which tools to use based on user input:

```python
class RestaurantAgent:
    def __init__(self, model):
        self.model = model
        self.context = ConversationContext()
    
    def process_input(self, user_input):
        # Get current context
        context = self.context.get_formatted_context()
        
        # Determine intent
        intent = find_intent(user_input, context)
        
        # Select appropriate tool based on intent
        if intent == "RECOMMEND_RESTAURANT":
            response = suggest_restaurant(user_input, context)
        elif intent == "MAKE_RESERVATION":
            response = make_reservation(user_input, context)
        else:
            # General conversation
            response = self.handle_general_inquiry(user_input, context)
        
        # Update context
        self.context.add_exchange(user_input, response)
        return response
    
    def handle_general_inquiry(self, user_input, context):
        prompt = f"""
        You are a helpful restaurant assistant.
        
        Conversation history:
        {context}
        
        User: {user_input}
        Assistant:
        """
        return self.model(prompt).strip()
```

The controller follows a simple pattern:
1. Analyze the user input to determine intent
2. Select the appropriate tool based on intent
3. Execute the tool and generate a response
4. Update conversation context


<img src="[https://github.com/user-attachments/assets/3a778e14-ce9f-43f5-8bd8-66101cb77358](https://github.com/user-attachments/assets/657e0596-2b09-46b2-aff5-4d1285643791)" alt="image" width="300">

### 5. Creating a Simple UI

For our example, we've created a basic chat interface:

```python
# Simple terminal-based UI example
def main():
    agent = RestaurantAgent(model)
    print("Restaurant Assistant: Hello! How can I help you find the perfect dining experience today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
            
        response = agent.process_input(user_input)
        print(f"Restaurant Assistant: {response}")

if __name__ == "__main__":
    main()
```

For web-based applications, you might implement a simple Flask or FastAPI backend with a JavaScript frontend.

## Advantages of Building Agents from Scratch

The Hugging Face Agents Course highlights several benefits of building from scratch rather than relying on frameworks:

1. **Complete Understanding**: You grasp every component's role and interaction
2. **Maximum Flexibility**: Customize each component to your exact requirements
3. **Lightweight Implementation**: Avoid bloated dependencies and overhead
4. **Easier Debugging**: Trace issues through your own clean, minimalist code
5. **Specific Optimizations**: Tune performance for your particular use case
6. **Custom Integrations**: Seamlessly connect with your existing tech stack
7. **Learning Opportunity**: Gain deeper insights into agent architecture fundamentals

## Challenges and Considerations

While building from scratch offers advantages, it comes with challenges:

1. **Prompt Engineering**: Crafting effective prompts requires experimentation
2. **Error Handling**: You need robust error handling for all components
3. **Scalability**: Consider how your agent will scale with increased usage
4. **Testing**: Build comprehensive tests for each component

## Extending Your Agent

Once you have the basic architecture, you can enhance your agent with:

1. **Structured Output Parsing**: Add JSON or structured text parsing
2. **Tool Chaining**: Allow tools to call other tools in sequence
3. **Long-Term Memory**: Implement database storage for persistent memory
4. **User Authentication**: Add user-specific contexts and preferences
5. **Multi-Modal Capabilities**: Extend to handle images or other input types

### Advanced Agent Patterns

As you evolve your agent architecture, consider these advanced patterns:

#### ReAct: Reasoning and Acting

The ReAct pattern combines reasoning and acting in an iterative loop:
```
1. Observe (receive input or tool results)
2. Think (reason about the current state)
3. Act (use a tool or generate a response)
4. Repeat until the task is complete
```

This pattern helps agents tackle complex problems by breaking them down into manageable steps and incorporating feedback at each stage.

#### Self-Reflection and Refinement

Advanced agents can benefit from self-reflection capabilities:

```python
def self_reflect(agent_response, user_query, context):
    prompt = f"""
    Review your proposed response:
    {agent_response}
    
    Based on the user query:
    {user_query}
    
    And conversation context:
    {context}
    
    Is this response complete, accurate, and helpful? 
    If not, provide an improved response:
    """
    return llm(prompt)
```

This allows agents to catch their own mistakes and improve responses before sending them to users.

#### Planning with Tool Composition

Complex tasks often require multiple tools used in sequence. A planning component can help:

```python
def create_plan(user_input, available_tools, context):
    prompt = f"""
    Based on the user request: {user_input}
    
    Create a step-by-step plan using these available tools:
    {available_tools}
    
    Return a JSON array of steps, each with:
    - tool_name: The tool to use
    - tool_input: The parameters to pass
    - description: Why this step is needed
    """
    plan = llm(prompt)
    return json.loads(plan)
```

This allows agents to decompose complex tasks into manageable sequences of tool calls.



## The Why: Advantages of Agents Over Traditional Approaches

Now that we understand the what and how of AI agents, let's explore why they represent such a compelling approach to building AI systems:

### 1. Complex Task Handling

Traditional conversational AI struggles with multi-step tasks that require memory, planning, and external actions. Agents excel precisely in these scenarios:

- **Sequential Decision Making**: Making reservations requires gathering multiple pieces of information in a logical sequence
- **Conditional Logic**: Different restaurant recommendations based on user preferences
- **Dynamic Responses**: Adjusting recommendations based on availability information

### 2. Specialized Domain Expertise

While general-purpose LLMs have broad knowledge, agents can focus on specific domains with:

- **Custom Tools**: Restaurant databases, reservation systems, cuisine-specific knowledge bases
- **Tailored Prompts**: Optimized for restaurant recommendations and bookings
- **Specialized Training**: Fine-tuning on restaurant industry data

### 3. Enhanced User Experience

Agents provide a more satisfying user experience by:

- **Reducing Friction**: Completing tasks without requiring users to switch between multiple systems
- **Providing Context Awareness**: Remembering user preferences from previous interactions
- **Supporting Iterative Refinement**: "No, I need something cheaper" is handled naturally

### 4. Scalability and Extensibility

A well-designed agent architecture can scale to:

- **More Complex Domains**: The same architecture can handle everything from restaurant bookings to travel planning
- **Additional Tools**: Easily add payment processing, loyalty programs, or dietary restriction checking
- **Multiple Channels**: Deploy the same agent across web, mobile, and voice interfaces

## Conclusion

Building AI agents from scratch provides invaluable insights into how these systems work while offering maximum flexibility. The restaurant agent example demonstrates that with relatively simple components—a language model, context management, tools, and a controller—you can create a functional AI agent without relying on complex frameworks.

As language models continue to improve, the potential applications for custom-built agents will only grow. By mastering the fundamentals of agent architecture, you're positioning yourself at the forefront of a transformative approach to AI application development.

For a complete implementation example, check out [ByteTrooper's AI-Agent repository](https://github.com/ByteTrooper/AI-Agent), which demonstrates these concepts in a working restaurant recommendation system.

By understanding these fundamentals, you'll be better equipped to build, customize, and debug AI agents for any application—whether you eventually choose to use frameworks or continue building from first principles.

## Future Directions in Agent Development

Looking ahead, several exciting trends are emerging in agent development, many of which are explored in the Hugging Face Agents Course:

1. **Multi-agent Systems**: Specialized agents collaborating to solve complex problems
2. **Agentic Memory Architectures**: More sophisticated approaches to long-term knowledge retention
3. **Tool Learning**: Agents that discover and learn to use new tools without explicit programming
4. **Autonomous Agent Communities**: Self-governing systems of agents that coordinate activities
5. **Human-Agent Collaboration**: Enhanced frameworks for humans and agents to work together effectively
6. **Reinforcement Learning from Human Feedback (RLHF)**: Improving agent behavior based on human preferences
7. **Cross-modal Agents**: Working with text, images, audio, and video simultaneously
8. **Embodied Agents**: Connecting AI agents to robotic systems for physical world interaction

The field is evolving rapidly, and by building your own agents from scratch, you'll have the foundational knowledge to adapt to and leverage these advances as they emerge.

## Learning Resources

If you're interested in diving deeper into AI agent development, here are some valuable resources:

1. [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) - A comprehensive introduction to agent concepts and implementation
2. [ByteTrooper's AI-Agent Repository](https://github.com/ByteTrooper/AI-Agent) - Our example restaurant agent implementation
3. LangChain and LlamaIndex documentation (for comparison with framework approaches)
4. Academic papers on agent architectures like ReAct, Reflexion, and MRKL

---

What agent will you build next? The possibilities are endless once you understand the core architecture!
