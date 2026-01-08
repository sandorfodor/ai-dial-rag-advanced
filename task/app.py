from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant specialized in microwave oven usage and safety.

Your responses are based on a specific microwave manual (DW 395 HCG model). Each user message will contain:
1. RAG Context - relevant information retrieved from the microwave manual
2. User Question - the actual question being asked

IMPORTANT INSTRUCTIONS:
- Use ONLY the information provided in the RAG Context to answer questions
- If the question cannot be answered from the provided context, politely state that you don't have that information in the manual
- Do not answer questions unrelated to microwave usage, operation, safety, or maintenance
- Do not answer questions outside the scope of the provided manual context
- Be accurate, helpful, and concise in your responses
- If the question is off-topic (e.g., about dinosaurs, other products, general knowledge), politely decline and remind the user of your specific purpose
"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """RAG Context:
{context}

---

User Question:
{question}
"""


#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

def main():
    """Main RAG application"""
    # Initialize clients
    embeddings_client = DialEmbeddingsClient(
        deployment_name='text-embedding-3-small-1',
        api_key=API_KEY
    )
    
    chat_client = DialChatCompletionClient(
        deployment_name='gpt-4o',
        api_key=API_KEY
    )
    
    # Initialize text processor with database configuration
    db_config = {
        'host': 'pgvector-db',
        'port': 5432,  # Internal container port, not the host-mapped port
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    text_processor = TextProcessor(embeddings_client, db_config)
    
    # Process the microwave manual (one-time setup)
    import os
    manual_path = os.path.join(os.path.dirname(__file__), 'embeddings', 'microwave_manual.txt')
    
    print("Processing microwave manual...")
    text_processor.process_text_file(
        file_path=manual_path,
        chunk_size=300,
        overlap=40,
        dimensions=1536,
        truncate_table=True
    )
    print("Manual processed successfully!\n")
    
    # Initialize conversation with system prompt
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    
    print("=" * 60)
    print("Microwave Manual RAG Assistant")
    print("=" * 60)
    print("Ask questions about the microwave manual.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Console chat loop
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # Step 1: Retrieval - Get relevant context from the manual
        relevant_chunks = text_processor.search(
            query=user_input,
            search_mode=SearchMode.COSINE_DISTANCE,
            top_k=5,
            min_score=0.5,
            dimensions=1536
        )
        
        # Step 2: Augmentation - Combine context with user question
        context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."
        augmented_prompt = USER_PROMPT.format(
            context=context,
            question=user_input
        )
        
        # Add user message to conversation
        user_message = Message(Role.USER, augmented_prompt)
        conversation.add_message(user_message)
        
        # Step 3: Generation - Get AI response
        try:
            ai_response = chat_client.get_completion(conversation.get_messages())
            conversation.add_message(ai_response)
            
            print(f"\nAssistant: {ai_response.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


main()


# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml