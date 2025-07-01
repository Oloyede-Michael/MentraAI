# MentraAI

MentraAI is an innovative, emotionally intelligent chatbot designed to provide compassionate mental health support and resources to individuals experiencing distress. By leveraging advanced language models and emotion analysis, MentraAI delivers empathetic and context-aware conversations that help users feel genuinely heard, understood, and supported.

In addition to its core mental health support capabilities, MentraAI also integrates a visual diagnostic feature. This allows users to upload images of visible ailments—such as pimples, rashes, or skin conditions like measles—for preliminary analysis. Utilizing cutting-edge image recognition technology, MentraAI can provide a diagnostic suggestion with up to 79% accuracy, helping users better understand their symptoms and make informed decisions about seeking professional care.

---

## Tech Stack

### Frontend Frameworks

- **React** (with **Vite**)
- **Tailwind CSS**
- **Next.js**

### Backend

- **Express.js**
- **MongoDB** (used for user authentication and storage)

---

## Features

- **Empathetic Chatbot:** Engages in supportive, emotionally-intelligent conversations.
- **Emotion Analysis:** Detects and responds to user emotions to provide tailored responses.
- **Dynamic Conversation Flow:** Adjusts conversation strategy based on user mood and conversation stage (e.g., greeting, support, crisis).
- **Configurable Behavior:** Users can tune empathy level, response style, and emotion sensitivity.
- **Supports Multimodal Inputs:** Can process text, and (in advanced configs) voice and visual content.
- **Document Ingestion:** Supports ingestion of various document types for resource enrichment.
- **Visual Diagnostics:** Users can upload images for preliminary analysis of visible ailments.

---

## Directory Structure

- `src/mentraai/` - Core implementation of MentraAI, including chatbot logic and emotion analysis.
- `src/mentraai.egg-info/` - Python package metadata.
- `aiqtoolkit-opensource-ui/` - (Optional) UI tools or components.
- `configs/` - Configuration files.
- `data/` - Example or production data.
- `mentra_landpage/` - Landing page resources.
- `pyproject.toml` - Python project configuration.

---

## How It Works

- **Emotion Detection:** User input is analyzed for primary and secondary emotions (anger, fear, sadness, etc.) along with emotional intensity.
- **Context Tracking:** Maintains conversation context, user mood history, and interaction counts to personalize responses.
- **Adaptive Response:** EmpathicChatbot dynamically generates responses based on current emotional and conversational context.
- **Crisis Handling:** Recognizes high-intensity negative emotions and escalates to crisis support if needed.
- **Visual Diagnostics:** Users can upload images for AI-powered preliminary health analysis.

---

## Setup

### Prerequisites

- Python 3.8+
- NVIDIA API key for LLM endpoints (set as `NVIDIA_API_KEY` in your environment)
- Node.js (for frontend and backend components)
- MongoDB instance (local or cloud)
- Required Python packages (see `pyproject.toml`)
- (Optional) `dotenv` for environment variable management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Oloyede-Michael/MentraAI.git
   cd MentraAI
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or, if using Poetry or PDM:
   ```bash
   poetry install
   # or
   pdm install
   ```

3. **Install Frontend and Backend dependencies:**
   ```bash
   # For React/Vite/Next.js frontend
   cd aiqtoolkit-opensource-ui
   npm install

   # For Express backend
   cd ../backend
   npm install
   ```

4. **Set environment variables:**
   - Create a `.env` file or export required variables:
     ```
     NVIDIA_API_KEY=your_nvidia_api_key
     MONGODB_URI=your_mongodb_connection_string
     ```

---

### Running the Application

- **Python Backend (Empathic AI):**
  - The core logic is in `src/mentraai/mentraai_function.py`. The chatbot can be integrated into a service, UI, or run as a standalone process (with custom CLI/web integration).

  Example (Python):
  ```python
  from mentraai.mentraai_function import EmpathicChatbot, MentraaiFunctionConfig, Builder

  config = MentraaiFunctionConfig(
      # configure as needed
  )
  builder = Builder()
  bot = EmpathicChatbot(config, builder)
  # Use bot.process_conversation(...) for chat
  ```

- **Frontend (React/Vite/Tailwind or Next.js):**
  - From the frontend directory:
    ```bash
    npm run dev
    ```
  - (See individual frontend `README.md` for further instructions.)

- **Node/Express Backend:**
  - From the backend directory:
    ```bash
    node index.js
    ```
    or
    ```bash
    npm run start
    ```

---

### Configuration

The chatbot is highly configurable. Key options include:

- `empathy_level`: "low", "medium", or "high"
- `response_style`: "supportive", "professional", "casual", "therapeutic"
- `emotion_sensitivity`: Float (0.1–1.0)
- `max_history`: Number of conversation history tokens to keep

Edit these settings in your initialization or configuration files.

---

## Contributing

1. Fork the repo and clone your fork.
2. Create a new branch: `git checkout -b my-feature`
3. Make your changes and commit: `git commit -am 'Add new feature'`
4. Push and open a Pull Request.

---

## License

This project currently does not specify a license. Please contact the repository owner for intended usage or contribution terms.

---

## Acknowledgments

MentraAI is built with:
- [Langchain](https://github.com/langchain-ai/langchain)
- NVIDIA AI endpoints
- Python ecosystem (Pydantic, dotenv, etc.)
- React, Vite, Tailwind CSS, Next.js
- Express.js, MongoDB

---

For questions, issues, or feature requests, please open an issue on [GitHub](https://github.com/Oloyede-Michael/MentraAI).
