import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image
import requests
from io import BytesIO
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import asyncio
import threading
import time
import re

# Load environment variables
load_dotenv()

set_tracing_disabled(disabled=True)

# Initialize Flask app
app = Flask(__name__)

# File paths
PRODUCTS_FILE = 'products.json'
AGENT_MEMORY_FILE = 'agent_memory.json'
CUSTOMER_DATA_FILE = 'customer_data.json'
ORDERS_FILE = 'orders.json'

def init_files():
    """Initialize all required JSON files if they don't exist"""
    files = {
        AGENT_MEMORY_FILE: {"product_analysis": {}, "image_analysis": {}},
        CUSTOMER_DATA_FILE: {"customers": []},
        ORDERS_FILE: {"orders": []}
    }
    
    for file_path, default_data in files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(default_data, f, indent=2)

def analyze_product_image(image_url: str) -> Dict[str, Any]:
    """Analyze product image using Pillow"""
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        analysis = {
            "size": img.size,
            "format": img.format,
            "mode": img.mode,
            "aspect_ratio": round(img.size[0] / img.size[1], 2)
        }
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get dominant colors
        img = img.resize((150, 150))
        colors = img.getcolors(150*150)
        if colors:
            colors.sort(reverse=True)
            analysis["dominant_colors"] = [color[1] for color in colors[:5]]
        
        return analysis
    except Exception as e:
        print(f"Error analyzing image {image_url}: {str(e)}")
        return {"error": str(e)}

def analyze_products():
    """Analyze products and store in agent memory"""
    try:
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        
        with open(AGENT_MEMORY_FILE, 'r') as f:
            memory = json.load(f)
        
        # Analyze each product
        for product in products_data["products"]:
            product_id = product["product_id"]
            
            # Skip if already analyzed
            if product_id in memory["product_analysis"]:
                continue
            
            # Analyze product
            analysis = {
                "category": product["type"],
                "price_range": "high" if product["price"] > 500 else "medium" if product["price"] > 100 else "low",
                "brand": product["brand"],
                "key_features": product["features"],
                "stock_status": "in_stock" if product["quantity_in_stock"] > 0 else "out_of_stock",
                "analyzed_at": datetime.now().isoformat()
            }
            
            # Analyze product image
            image_analysis = analyze_product_image(product["image_url"])
            memory["image_analysis"][product_id] = image_analysis
            
            memory["product_analysis"][product_id] = analysis
        
        # Save updated memory
        with open(AGENT_MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
            
    except Exception as e:
        print(f"Error analyzing products: {str(e)}")

# --- Helper functions for agent tools ---
def _get_customer_data(email: str) -> Optional[Dict[str, Any]]:
    try:
        with open(CUSTOMER_DATA_FILE, 'r') as f:
            data = json.load(f)
        for customer in data["customers"]:
            if customer["email"] == email:
                return customer
        return None
    except Exception as e:
        print(f"Error getting customer data: {str(e)}")
        return None

@function_tool
def get_customer_data(email: str) -> Optional[Dict[str, Any]]:
    return _get_customer_data(email)


def _save_customer_data(name: str, email: str) -> Dict[str, Any]:
    try:
        with open(CUSTOMER_DATA_FILE, 'r') as f:
            data = json.load(f)
        customer = {
            "name": name,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "conversation_history": []
        }
        data["customers"].append(customer)
        with open(CUSTOMER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return customer
    except Exception as e:
        print(f"Error saving customer data: {str(e)}")
        return None

@function_tool
def save_customer_data(name: str, email: str) -> Dict[str, Any]:
    return _save_customer_data(name, email)


def _update_conversation_history(email: str, message: str, response: str):
    try:
        with open(CUSTOMER_DATA_FILE, 'r') as f:
            data = json.load(f)
        for customer in data["customers"]:
            if customer["email"] == email:
                customer["conversation_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "response": response
                })
                break
        with open(CUSTOMER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating conversation history: {str(e)}")

@function_tool
def update_conversation_history(email: str, message: str, response: str):
    return _update_conversation_history(email, message, response)


def _get_relevant_products(query: str) -> List[Dict[str, Any]]:
    try:
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        with open(AGENT_MEMORY_FILE, 'r') as f:
            memory = json.load(f)
        relevant_products = []
        query = query.lower()
        for product in products_data["products"]:
            product_id = product["product_id"]
            analysis = memory["product_analysis"].get(product_id, {})
            if (query in product["name"].lower() or
                query in product["description"].lower() or
                query in product["type"].lower() or
                query in product["brand"].lower() or
                any(query in feature.lower() for feature in product["features"])):
                product["analysis"] = analysis
                product["image_analysis"] = memory["image_analysis"].get(product_id, {})
                relevant_products.append(product)
        return relevant_products
    except Exception as e:
        print(f"Error getting relevant products: {str(e)}")
        return []

@function_tool
def get_relevant_products(query: str) -> List[Dict[str, Any]]:
    return _get_relevant_products(query)


def _create_order(customer_email: str, product_id: str) -> Dict[str, Any]:
    try:
        customer = _get_customer_data(customer_email)
        if not customer:
            return {"error": "Customer not found"}
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        product = None
        for p in products_data["products"]:
            if p["product_id"] == product_id:
                product = p
                break
        if not product:
            return {"error": "Product not found"}
        order = {
            "order_id": f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "customer_email": customer_email,
            "customer_name": customer["name"],
            "product_id": product_id,
            "product_name": product["name"],
            "price": product["price"],
            "order_date": datetime.now().isoformat(),
            "status": "pending"
        }
        with open(ORDERS_FILE, 'r') as f:
            orders_data = json.load(f)
        orders_data["orders"].append(order)
        with open(ORDERS_FILE, 'w') as f:
            json.dump(orders_data, f, indent=2)
        return order
    except Exception as e:
        print(f"Error creating order: {str(e)}")
        return {"error": str(e)}

@function_tool
def create_order(customer_email: str, product_id: str) -> Dict[str, Any]:
    return _create_order(customer_email, product_id)

def extract_name_email(text):
    # Simple regex for email
    email_match = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text)
    email = email_match.group(1) if email_match else None

    # Try to extract name (very basic, you can improve)
    name = None
    if email:
        # Remove email from text, then look for 'name is ...' or 'I am ...'
        text_wo_email = text.replace(email, '')
        name_match = re.search(r'(?:name is|I am|I\'m|this is)\\s*([a-zA-Z ]+)', text_wo_email, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
    return name, email

class CustomerSupportAgent:
    def __init__(self):
        init_files()
        analyze_products()
        load_dotenv()
        
        # Twilio configuration
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER')
        self.recipient_number = os.getenv('RECIPIENT_WHATSAPP_NUMBER')
        
        # Initialize Twilio client
        self.client = Client(self.account_sid, self.auth_token)
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Create the agent
        self.agent = Agent(
            name="CustomerSupportAgent",
            instructions="""You are a professional customer support agent for an e-commerce store. Your responsibilities include:

1. Greeting new customers and collecting their name and email
2. Understanding customer needs and recommending relevant products
3. Providing detailed product information including images
4. Handling order confirmations
5. Maintaining conversation context and history
6. Being polite, professional, and helpful at all times

When recommending products:
- Only suggest products that match the customer's needs
- Include product images and key features
- Be honest about product availability and pricing
- Provide clear next steps for ordering

Remember to:
- Always verify customer identity
- Keep track of conversation history
- Be proactive in understanding customer needs
- Provide clear and concise responses
- Use proper formatting for better readability""",
            model=OpenAIChatCompletionsModel(
                model="gemini-2.0-flash",
                openai_client=self.openai_client
            ),
            tools=[
                get_customer_data,
                save_customer_data,
                update_conversation_history,
                get_relevant_products,
                create_order
            ]
        )
        
        print("\nCustomer Support Agent Configuration:")
        print(f"Twilio Account SID: {self.account_sid}")
        print(f"Twilio WhatsApp Number: {self.whatsapp_number}")
        print(f"Recipient Number: {self.recipient_number}")
        print("\nChecking WhatsApp configuration...")
        self.test_whatsapp_connection()

    def test_whatsapp_connection(self):
        try:
            message = self.client.messages.create(
                from_=f"whatsapp:{self.whatsapp_number}",
                body="Customer Support Agent is now active! How can I help you today?",
                to=f"whatsapp:{self.recipient_number}"
            )
            print("Successfully sent test WhatsApp message!")
        except Exception as e:
            print(f"Error testing WhatsApp connection: {str(e)}")
            print("Please ensure:")
            print("1. Your Twilio credentials are correct")
            print("2. Your WhatsApp numbers are properly formatted")
            print("3. You have sufficient credits in your Twilio account")

    def send_whatsapp_message(self, to_number: str, message: str):
        try:
            self.client.messages.create(
                body=message,
                from_=f"whatsapp:{self.whatsapp_number}",
                to=f"whatsapp:{to_number}"
            )
            print(f"WhatsApp message sent successfully to {to_number}!")
        except Exception as e:
            print(f"Failed to send WhatsApp message: {str(e)}")

    async def handle_message(self, message: str, sender: str) -> str:
        try:
            customer = _get_customer_data(sender)
            if not customer:
                # Try to extract name and email from the message
                name, email = extract_name_email(message)
                if name and email:
                    _save_customer_data(name, email)
                    return f"Thank you, {name}! You are now registered. How can I assist you today?"
                else:
                    return "Welcome! I'm your customer support assistant. Could you please provide your name and email address?"
            
            # Process the message using the agent
            result = await Runner.run(
                self.agent,
                message,
                context={
                    "customer_email": sender,
                    "customer_name": customer["name"],
                    "conversation_history": customer["conversation_history"]
                }
            )
            
            # Update conversation history
            _update_conversation_history(sender, message, result.final_output)
            
            return result.final_output
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Create a global instance of the agent
agent = CustomerSupportAgent()

@app.route("/webhook", methods=['POST'])
def webhook():
    # Get the message from the request
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '').replace('whatsapp:', '')
    
    try:
        # Process the message using the agent
        response = asyncio.run(agent.handle_message(incoming_msg, sender))
        
        # Send the response back via WhatsApp
        agent.send_whatsapp_message(sender, response)
        
        # Return empty TwiML response
        twiml = MessagingResponse()
        return str(twiml)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        agent.send_whatsapp_message(sender, error_msg)
        twiml = MessagingResponse()
        return str(twiml)

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def main():
    required_vars = [
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN',
        'TWILIO_WHATSAPP_NUMBER',
        'RECIPIENT_WHATSAPP_NUMBER',
        'GEMINI_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    print("Customer Support Agent started...")
    print("Webhook server running on http://0.0.0.0:5000/webhook")
    print("\nYou can now interact with the agent through WhatsApp!")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCustomer Support Agent stopped.")

if __name__ == "__main__":
    main()
