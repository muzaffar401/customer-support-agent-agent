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
import asyncio
import re

# Load environment variables
load_dotenv()

set_tracing_disabled(disabled=True)

# File paths
PRODUCTS_FILE = 'products.json'
AGENT_MEMORY_FILE = 'agent_memory.json'
CUSTOMER_DATA_FILE = 'customer_data.json'
ORDERS_FILE = 'orders.json'

# Load products data
with open(PRODUCTS_FILE, 'r') as f:
    products_data = json.load(f)

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
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        # Try to download the image with headers and timeout
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        analysis = {
            "size": img.size,
            "format": img.format,
            "mode": img.mode,
            "aspect_ratio": round(img.size[0] / img.size[1], 2) if img.size[1] > 0 else 0,
            "status": "success",
            "url": image_url
        }
        
        try:
            # Convert to RGB if needed
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            
            # Get dominant colors
            img = img.resize((150, 150))  # Resize for faster processing
            colors = img.getcolors(150*150)
            if colors:
                colors.sort(reverse=True)
                analysis["dominant_colors"] = [color[1] for color in colors[:5]]
        except Exception as color_error:
            print(f"Warning: Could not analyze colors for {image_url}: {str(color_error)}")
            analysis["dominant_colors"] = []
            analysis["color_error"] = str(color_error)
        
        return analysis
        
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 403:
            print(f"Warning: Access forbidden to image {image_url}. The website may be blocking automated access.")
            return {
                "status": "error",
                "error_type": "access_forbidden",
                "message": "Image access restricted by the server",
                "url": image_url
            }
        else:
            print(f"Warning: HTTP error occurred while accessing {image_url}: {str(http_err)}")
            return {
                "status": "error",
                "error_type": "http_error",
                "message": str(http_err),
                "url": image_url
            }
    except requests.exceptions.RequestException as req_err:
        print(f"Warning: Network error occurred while accessing {image_url}: {str(req_err)}")
        return {
            "status": "error",
            "error_type": "network_error",
            "message": str(req_err),
            "url": image_url
        }
    except Exception as e:
        print(f"Warning: Could not analyze image {image_url}: {str(e)}")
        return {
            "status": "error",
            "error_type": "processing_error",
            "message": str(e),
            "url": image_url
        }

def analyze_products():
    """Analyze products and store in agent memory"""
    try:
        # Load products and memory
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        
        with open(AGENT_MEMORY_FILE, 'r') as f:
            memory = json.load(f)
        
        # Initialize memory sections if they don't exist
        if "product_analysis" not in memory:
            memory["product_analysis"] = {}
        if "image_analysis" not in memory:
            memory["image_analysis"] = {}
        
        products_analyzed = 0
        images_failed = 0
        
        # Analyze each product
        for product in products_data["products"]:
            product_id = product["product_id"]
            
            # Skip if already analyzed and analysis is complete
            if (product_id in memory["product_analysis"] and 
                product_id in memory["image_analysis"] and 
                memory["product_analysis"][product_id].get("analyzed_at")):
                continue
            
            print(f"Analyzing product: {product['name']}")
            
            # Analyze product details
            analysis = {
                "category": product.get("type", ""),
                "price_range": "high" if float(product.get("price", 0)) > 500 else 
                              "medium" if float(product.get("price", 0)) > 100 else "low",
                "brand": product.get("brand", ""),
                "key_features": product.get("features", []),
                "stock_status": "in_stock" if int(product.get("quantity_in_stock", 0)) > 0 else "out_of_stock",
                "analyzed_at": datetime.now().isoformat()
            }
            
            # Analyze product image
            if "image_url" in product:
                image_analysis = analyze_product_image(product["image_url"])
                memory["image_analysis"][product_id] = image_analysis
                
                # Count failed images
                if image_analysis.get("status") == "error":
                    images_failed += 1
                    # Add a note about the image issue to the product analysis
                    analysis["image_status"] = f"Image analysis failed: {image_analysis.get('message', 'Unknown error')}"
                else:
                    analysis["image_status"] = "Image analysis successful"
            else:
                analysis["image_status"] = "No image URL provided"
                memory["image_analysis"][product_id] = {"status": "error", "error_type": "no_url", "message": "No image URL provided"}
            
            memory["product_analysis"][product_id] = analysis
            products_analyzed += 1
        
        # Save updated memory
        with open(AGENT_MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
            
        print(f"\nProduct analysis complete:")
        print(f"- Analyzed {products_analyzed} new products")
        if images_failed > 0:
            print(f"- {images_failed} images could not be analyzed (see warnings above)")
        print(f"- All product data has been saved to memory")
            
    except Exception as e:
        print(f"Error in analyze_products: {str(e)}")
        raise

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
        # Check if customer already exists, update name if needed
        for customer in data["customers"]:
            if customer["email"] == email:
                customer["name"] = name
                break
        else:
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


def _update_conversation_history(email: str, message: str, response: str, context_data: str = None):
    """Update conversation history with message, response and context"""
    try:
        with open(CUSTOMER_DATA_FILE, 'r') as f:
            data = json.load(f)
        for customer in data["customers"]:
            if customer["email"] == email:
                # Create conversation entry
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "response": response
                }
                # Add context if provided
                if context_data:
                    try:
                        context = json.loads(context_data)
                        entry.update({"context": context})
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse context data: {context_data}")
                customer["conversation_history"].append(entry)
                break
        with open(CUSTOMER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating conversation history: {str(e)}")

@function_tool
def update_conversation_history(email: str, message: str, response: str, context_data: str = None):
    """Update conversation history with message, response and context.
    
    Args:
        email: Customer's email address
        message: Customer's message
        response: Agent's response
        context_data: JSON string containing conversation context
    """
    return _update_conversation_history(email, message, response, context_data)


def _get_relevant_products(query: str) -> List[Dict[str, Any]]:
    """Get relevant products based on search query"""
    try:
        # Load products and memory
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        with open(AGENT_MEMORY_FILE, 'r') as f:
            memory = json.load(f)
        
        relevant_products = []
        query = query.lower()
        
        # Search through products
        for product in products_data["products"]:
            product_id = product["product_id"]
            
            # Create searchable text from all product fields
            searchable_text = " ".join([
                product["name"].lower(),
                product["description"].lower(),
                product["type"].lower(),
                product.get("brand", "").lower(),
                *[str(feature).lower() for feature in product["features"]]
            ])
            
            # Check for match in any field
            if any(term.lower() in searchable_text for term in query.split()):
                # Add memory data
                if product_id in memory["product_analysis"]:
                    product["analysis"] = memory["product_analysis"][product_id]
                if product_id in memory["image_analysis"]:
                    product["image_analysis"] = memory["image_analysis"][product_id]
                relevant_products.append(product)
        
        return relevant_products
    except Exception as e:
        print(f"Error in _get_relevant_products: {str(e)}")
        return []

@function_tool
def get_relevant_products(query: str) -> List[Dict[str, Any]]:
    return _get_relevant_products(query)


def _create_order(customer_email: str, product_id: str) -> Dict[str, Any]:
    """Create a new order for a customer"""
    try:
        # Get customer data
        customer = _get_customer_data(customer_email)
        if not customer:
            return {"error": "Customer not found. Please provide your name and email."}

        # Get product data
        with open(PRODUCTS_FILE, 'r') as f:
            products_data = json.load(f)
        
        # Find the product
        product = None
        for p in products_data["products"]:
            if p["product_id"] == product_id:
                product = p
                break
        
        if not product:
            return {"error": f"Product with ID {product_id} not found"}
        
        if product["quantity_in_stock"] <= 0:
            return {"error": f"Sorry, {product['name']} is currently out of stock"}

        # Create the order
        order = {
            "order_id": f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "customer_email": customer_email,
            "customer_name": customer["name"],
            "product_id": product_id,
            "product_name": product["name"],
            "price": product["price"],
            "order_date": datetime.now().isoformat(),
            "status": "confirmed"
        }

        # Update orders file
        try:
            with open(ORDERS_FILE, 'r') as f:
                orders_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            orders_data = {"orders": []}

        orders_data["orders"].append(order)
        
        with open(ORDERS_FILE, 'w') as f:
            json.dump(orders_data, f, indent=2)

        # Update product stock
        product["quantity_in_stock"] -= 1
        with open(PRODUCTS_FILE, 'w') as f:
            json.dump(products_data, f, indent=2)

        return {
            "success": True,
            "order": order,
            "message": f"Order created successfully! Order ID: {order['order_id']}"
        }
        
    except Exception as e:
        print(f"Error creating order: {str(e)}")
        return {"error": "An error occurred while creating your order"}

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

def is_sensitive_query(message: str) -> bool:
    """Check if the query contains sensitive information requests"""
    sensitive_keywords = [
        "inventory count", "total inventory", "database", "customer data", "customer list",
        "sales data", "revenue", "profit", "margin", "password", "credentials",
        "admin", "backend", "server", "system info", "internal", "employee", "staff",
        "security", "vulnerability", "exploit", "hack", "breach", "access",
        "sql", "query", "dump", "backup", "log", "config"
    ]
    # Use more specific matching to avoid false positives
    message_lower = message.lower()
    return any(keyword in message_lower and not any(safe_word in message_lower for safe_word in 
              ["looking for", "search", "find", "want", "show me", "tell me about", "price"]) 
              for keyword in sensitive_keywords)

def is_valid_product_query(message: str) -> bool:
    """Check if the query is a valid product-related question"""
    product_keywords = [
        "price", "cost", "feature", "specification", "detail", "color", "size",
        "brand", "model", "warranty", "shipping", "delivery", "availability",
        "stock", "order", "buy", "purchase", "return", "refund", "discount",
        "sale", "offer", "deal", "compare", "difference", "similar", "alternative",
        "recommend", "suggest", "review", "rating", "feedback", "quality",
        "looking for", "search", "find", "want", "show me", "tell me about",
        "how much", "what is", "do you have"
    ]
    # Also check for product categories or common product types
    product_categories = [
        "shoe", "shoes", "laptop", "phone", "computer", "watch", "clothing",
        "electronics", "book", "game", "furniture", "accessory", "accessories"
    ]
    message_lower = message.lower()
    return (any(keyword in message_lower for keyword in product_keywords) or
            any(category in message_lower for category in product_categories))

def is_personal_info_request(message: str) -> bool:
    """Check if the query is requesting personal information"""
    # Only match specific patterns for personal info requests
    personal_info_patterns = [
        r"show.*my.*address",
        r"what.*my.*phone",
        r"tell.*my.*email",
        r"get.*my.*credit card",
        r"access.*my.*payment",
        r"what.*my.*social security",
        r"show.*my.*identity",
        r"tell.*my.*birthday",
        r"what.*my.*age",
        r"show.*my.*order history",
        r"access.*my.*account",
        r"what.*my.*password"
    ]
    message_lower = message.lower()
    
    # Allow queries about the customer's own name or basic info
    if any(phrase in message_lower for phrase in ["my name", "who am i", "what's my name"]):
        return False
        
    return any(re.search(pattern, message_lower) for pattern in personal_info_patterns)

def get_safe_response(query_type: str) -> str:
    """Get appropriate response for different types of invalid queries"""
    responses = {
        "sensitive": "I apologize, but I cannot provide information about internal systems or sensitive business data. How can I help you with our products today?",
        "personal": "For security reasons, I can only provide basic information about your current session. For detailed account information, please use our secure customer portal.",
        "irrelevant": "I'm here to help you with our products and services. Could you please ask about specific products, pricing, or place an order?",
        "unclear": "I'm not sure I understand. Could you please ask about specific products or services?"
    }
    return responses.get(query_type, responses["unclear"])

class CustomerSupportAgent:
    def __init__(self):
        init_files()
        # Ensure agent memory is populated before creating the agent
        if not self._is_agent_memory_populated():
            print("Analyzing products and saving to agent memory...")
            analyze_products()
            print("Product analysis complete.")
        else:
            print("Product analysis already exists in agent memory.")
        load_dotenv()
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        # Create the agent
        self.agent = Agent(
            name="CustomerSupportAgent",
            instructions="""
You are a professional customer support agent for an e-commerce store. Your responsibilities include:

1. Greeting new customers and collecting their name and email
2. Understanding customer needs and recommending relevant products
3. Providing detailed product information including images
4. Handling order confirmations
5. Maintaining conversation context and history
6. Being polite, professional, and helpful at all times

IMPORTANT:
- If the customer's name and email are already provided in the context, DO NOT ask for them again.
- Only ask for name/email if they are missing from the context.
- Use the provided context to personalize your responses.

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
- Use proper formatting for better readability
""",
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
        print("\nTwilio integration removed. Running in local mode.")
        print("\nYou can now interact with the agent through HTTP POST requests!")

    def _is_agent_memory_populated(self):
        try:
            with open(AGENT_MEMORY_FILE, 'r') as f:
                memory = json.load(f)
            return bool(memory.get("product_analysis")) and bool(memory.get("image_analysis"))
        except Exception:
            return False

    async def handle_message(self, message: str, sender: str) -> str:
        try:
            # Get customer data first
            customer = _get_customer_data(sender)
            
            # Handle registration/identification
            if not customer:
                if "name is" not in message.lower() and "email is" not in message.lower():
                    return "You are not registered. Please provide your name and email to continue."

            # Initialize product tracking
            current_product = None
            current_product_name = None
            
            # Extract specific product mentions from current message
            message_lower = message.lower()
            
            # Create a mapping of product names to their IDs for easier lookup
            product_name_map = {product["name"].lower(): product["product_id"] for product in products_data["products"]}
            product_keywords_map = {
                "nike": "Nike Air Max",
                "adidas": "Adidas Ultraboost 22",
                "macbook": "MacBook Pro 2023",
                "ipad": "iPad Pro 12.9-inch",
                # Add more product keyword mappings as needed
            }
            
            # Check for product mentions in current message
            for keyword, full_name in product_keywords_map.items():
                if keyword in message_lower:
                    current_product = product_name_map.get(full_name.lower())
                    current_product_name = full_name
                    break
            
            # If no specific product found in current message, check conversation history
            if not current_product and customer and customer["conversation_history"]:
                last_product_context = None
                
                # Look through conversation history in reverse
                for conv in reversed(customer["conversation_history"]):
                    # Skip order confirmations to avoid reordering old products
                    if "order has been confirmed" in conv.get("response", "").lower():
                        break
                        
                    # Check context for product information
                    context = conv.get("context", {})
                    if context.get("product_id") and context.get("product_name"):
                        last_product_context = context
                        break
                    
                    # Check response for product mentions
                    response_lower = conv.get("response", "").lower()
                    for product in products_data["products"]:
                        if product["name"].lower() in response_lower:
                            last_product_context = {
                                "product_id": product["product_id"],
                                "product_name": product["name"]
                            }
                            break
                    if last_product_context:
                        break
                
                if last_product_context:
                    current_product = last_product_context["product_id"]
                    current_product_name = last_product_context["product_name"]

            # Handle order requests
            order_keywords = ["order", "buy", "purchase", "get it", "confirm"]
            is_order_request = any(keyword in message_lower for keyword in order_keywords)
            
            if is_order_request:
                # If we have a current product from the message or history
                if current_product:
                    # Verify it's the right product before ordering
                    product_info = next((p for p in products_data["products"] if p["product_id"] == current_product), None)
                    if product_info:
                        # Double check if user specified a different product
                        for keyword, full_name in product_keywords_map.items():
                            if keyword in message_lower and full_name.lower() != product_info["name"].lower():
                                # User wants a different product
                                for p in products_data["products"]:
                                    if p["name"].lower() == full_name.lower():
                                        current_product = p["product_id"]
                                        current_product_name = p["name"]
                                        product_info = p
                                        break
                        
                        # Create the order
                        order_result = _create_order(sender, current_product)
                        if "error" in order_result:
                            return f"Sorry, I couldn't create your order: {order_result['error']}"
                        elif "success" in order_result:
                            response = (f"Great news! Your order has been confirmed:\n\n"
                                      f"Order ID: {order_result['order']['order_id']}\n"
                                      f"Product: {order_result['order']['product_name']}\n"
                                      f"Price: ${order_result['order']['price']}\n"
                                      f"Status: {order_result['order']['status']}\n\n"
                                      f"Thank you for your purchase!")
                            
                            context_data = {
                                "product_id": current_product,
                                "product_name": current_product_name,
                                "is_order_request": True,
                                "is_order_confirmed": True,
                                "order_id": order_result['order']['order_id']
                            }
                            _update_conversation_history(sender, message, response, json.dumps(context_data))
                            return response
                else:
                    # No product specified, check if one was recently discussed
                    relevant_products = []
                    if customer and customer["conversation_history"]:
                        for conv in reversed(customer["conversation_history"]):
                            if "response" in conv and any(p["name"].lower() in conv["response"].lower() 
                                                        for p in products_data["products"]):
                                for product in products_data["products"]:
                                    if product["name"].lower() in conv["response"].lower():
                                        relevant_products.append(product)
                    
                    if len(relevant_products) == 1:
                        return f"Would you like to order the {relevant_products[0]['name']}? Please confirm by saying 'yes' or specify a different product."
                    elif len(relevant_products) > 1:
                        response = "I found multiple products in our conversation. Which one would you like to order?\n\n"
                        for idx, product in enumerate(relevant_products, 1):
                            response += f"{idx}. {product['name']} (${product['price']})\n"
                        return response
                    else:
                        return "Could you please specify which product you would like to order?"

            # Handle product queries
            product_keywords = ["looking for", "search", "find", "want", "show me", "tell me", "price of", "what is", "how much"]
            is_product_query = any(keyword in message_lower for keyword in product_keywords)
            
            # Get relevant products for the query
            relevant_products = []
            if is_product_query:
                # First try to find products mentioned in current message
                for product in products_data["products"]:
                    product_name_lower = product["name"].lower()
                    if product_name_lower in message_lower or any(word in product_name_lower for word in message_lower.split()):
                        relevant_products.append(product)
                
                # If no products found, do a broader search
                if not relevant_products:
                    relevant_products = _get_relevant_products(message)

            # Create context for the agent
            context = {
                "customer_email": customer["email"] if customer else None,
                "customer_name": customer["name"] if customer else None,
                "conversation_history": customer["conversation_history"] if customer else [],
                "current_product": current_product,
                "current_product_name": current_product_name,
                "relevant_products": relevant_products if relevant_products else None
            }

            # Process the message using the agent
            result = await Runner.run(
                self.agent,
                message,
                context=context
            )

            response = result.final_output
            
            # Store context with the conversation
            context_data = {
                "product_id": current_product,
                "product_name": current_product_name,
                "is_order_request": is_order_request,
                "is_product_query": is_product_query,
                "relevant_products": [p["product_id"] for p in relevant_products] if relevant_products else []
            }
            _update_conversation_history(sender, message, response, json.dumps(context_data))
            
            return response
            
        except Exception as e:
            error_msg = "I apologize, but I encountered an error. Please try asking your question differently."
            print(f"Error in handle_message: {str(e)}")  # Log the actual error but don't show to user
            return error_msg

# Create a global instance of the agent
agent = CustomerSupportAgent()

def main():
    required_vars = [
        'GEMINI_API_KEY'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return

    print("Customer Support Agent CLI started...")
    print("Type 'exit' to quit.")
    name = input("Enter your name: ").strip()
    email = input("Enter your email (for session memory): ").strip()
    if not email:
        print("No email provided. Exiting.")
        return

    # Save or update the customer at the start of the session
    _save_customer_data(name, email)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = asyncio.run(agent.handle_message(user_input, email))
        print(f"Agent: {response}")

if __name__ == "__main__":
    main()
