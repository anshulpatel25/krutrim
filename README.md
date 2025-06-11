# कृत्रिम

कृत्रिम (Krutrim) is a simple AI chatbot built using Streamlit, LangGraph, LangChain, and the Llama3.1 model. It is designed to provide helpful responses to user queries.

## Features

- Interactive chatbot interface powered by Streamlit.
- Uses LangGraph for state management and LangChain for language model integration.
- Powered by the Llama3.1 model via Ollama.

## Installation

Follow these steps to set up and run the project:

### Prerequisites

- Python 3.13 or higher
- [Pipenv](https://pipenv.pypa.io/en/latest/) for dependency management

### Steps

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/krutrim.git
   cd krutrim
   ```

2. Install dependencies:

   ```sh
   uv pip install -r pyproject.toml
   ```

3. Install Ollama:

   Ollama is required to run the Llama3.1 model. Follow the instructions below to install it:

   Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

4. Download the `llama3.1` model:

   Once Ollama is installed, run the following command to download the `llama3.1` model:

   ```sh
   ollama pull llama3.1
   ```

5. Run the application:

   Activate the Pipenv shell and start the Streamlit app:

   ```sh
   uv run streamlit run krutrim.py
   ```

6. Open your browser:

   The application will be available at `http://localhost:8501`.

## Usage

- Enter your query in the chatbot input field.
- The chatbot will respond with helpful answers.
- If the chatbot doesn't know the answer, it will respond with "I don't know."

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangGraph](https://github.com/langgraph/langgraph)
- [LangChain](https://langchain.com/)
- [Ollama](https://ollama.com/)
- [Llama3.1](https://ollama.com/library/llama3.1)
