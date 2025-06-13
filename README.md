# Customer Support Agent

A WhatsApp-based customer support agent that helps customers browse and purchase products through natural conversation.

## Features

- Product analysis and recommendation
- Image analysis using Pillow
- Customer data management
- Conversation history tracking
- Order management
- WhatsApp integration via Twilio
- Persistent memory for product and customer data
- ngrok integration for webhook testing

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-support-agent
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=your_twilio_whatsapp_number
RECIPIENT_WHATSAPP_NUMBER=your_whatsapp_number
GEMINI_API_KEY=your_gemini_api_key
NGROK_AUTH_TOKEN=your_ngrok_auth_token  # Optional but recommended
```

5. Ensure you have a `products.json` file with your product catalog.

## Running the Agent

1. Start the agent:
```bash
python main.py
```

2. The agent will:
   - Initialize required JSON files
   - Analyze products and store in memory
   - Start the Flask webhook server
   - Create an ngrok tunnel (if configured)
   - Send a test message to verify WhatsApp connection

3. The webhook server will be running at:
   - Local: `http://0.0.0.0:5000/webhook`
   - Public (via ngrok): `https://<your-ngrok-url>/webhook`

4. Update your Twilio webhook URL:
   - Go to your Twilio console
   - Update the webhook URL to the ngrok URL shown in the console
   - Make sure to append `/webhook` to the URL

## Usage

1. Send a message to the WhatsApp number
2. For new customers:
   - The agent will ask for name and email
   - After registration, you can start browsing products
3. For returning customers:
   - The agent will recognize you and load your history
   - You can continue browsing and ordering products

## File Structure

- `main.py`: Main application code
- `products.json`: Product catalog
- `agent_memory.json`: Agent's product analysis memory
- `customer_data.json`: Customer information and conversation history
- `orders.json`: Order records
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)

## Notes

- The agent uses Gemini for natural language processing
- Product images are analyzed using Pillow
- All data is stored in JSON files for persistence
- The agent maintains conversation context for better customer experience
- ngrok is used to expose the local server to the internet for Twilio webhooks
