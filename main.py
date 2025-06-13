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
import webbrowser
from fpdf import FPDF
import stripe
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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


def _save_customer_data(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        with open(CUSTOMER_DATA_FILE, 'r') as f:
            data = json.load(f)
        # Check if customer already exists, update if needed
        for customer in data["customers"]:
            if customer["email"] == customer_data["email"]:
                # Update existing customer data
                customer.update(customer_data)
                break
        else:
            # Add new customer
            data["customers"].append(customer_data)
        with open(CUSTOMER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return customer_data
    except Exception as e:
        print(f"Error saving customer data: {str(e)}")
        return None

@function_tool
def save_customer_data(name: str, email: str, phone: str, address: str, city: str, state: str, country: str, postal_code: str) -> Dict[str, Any]:
    """Save or update customer data with comprehensive details.
    
    Args:
        name: Customer's full name
        email: Customer's email address
        phone: Customer's phone number
        address: Customer's street address
        city: Customer's city
        state: Customer's state/province
        country: Customer's country
        postal_code: Customer's postal/zip code
    """
    customer_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "address": {
            "street": address,
            "city": city,
            "state": state,
            "country": country,
            "postal_code": postal_code
        },
        "created_at": datetime.now().isoformat(),
        "conversation_history": []
    }
    return _save_customer_data(customer_data)


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


def _create_order(customer_email: str, product_id: str, payment_method: str = "cod") -> Dict[str, Any]:
    """Create a new order for a customer in the new nested format"""
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

        # Prepare customer info with complete details
        customer_info = {
            "name": customer.get("name", "N/A"),
            "email": customer.get("email", "N/A"),
            "phone": customer.get("phone", "N/A"),
            "address": customer.get("address", {
                "street": "N/A",
                "city": "N/A",
                "state": "N/A",
                "country": "N/A",
                "postal_code": "N/A"
            })
        }
        
        # Prepare items list with image URL
        items = [{
            "name": product["name"],
            "quantity": 1,
            "unit_price": float(product["price"]),
            "image_url": product.get("image_url", "")
        }]
        
        # Calculate charges
        subtotal = float(product["price"])
        tax_rate = 0.01  # 1% tax
        tax = round(subtotal * tax_rate, 2)
        shipping_fee = 500  # Base shipping fee
        handling_fee = 100  # Handling fee
        
        # Add COD fee if payment method is COD
        cod_fee = 50 if payment_method.lower() == "cod" else 0
        
        # Calculate total
        total = subtotal + tax + shipping_fee + handling_fee + cod_fee
        
        # Create order with all details
        order = {
            "order_id": f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "order_date": datetime.now().isoformat(),
            "customer": customer_info,
            "items": items,
            "payment_method": payment_method.upper(),
            "charges": {
                "subtotal": subtotal,
                "tax_rate": f"{tax_rate * 100}%",
                "tax": tax,
                "shipping_fee": shipping_fee,
                "handling_fee": handling_fee,
                "cod_fee": cod_fee if payment_method.lower() == "cod" else 0,
                "total": total
            },
            "status": "pending",
            "payment_status": "pending"
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
def create_order(customer_email: str, product_id: str, payment_method: str = "cod") -> Dict[str, Any]:
    """Create a new order for a customer.
    Args:
        customer_email: The email of the customer placing the order.
        product_id: The ID of the product being ordered.
        payment_method: The payment method for the order (e.g., "cod", "credit_card"). Defaults to "cod".
    """
    return _create_order(customer_email, product_id, payment_method)

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
                        
                        try:
                            # Create the order
                            order_payment_method = "cod"
                            if "credit card" in message_lower or "card" in message_lower:
                                order_payment_method = "credit_card"
                            
                            order_result = _create_order(sender, current_product, payment_method=order_payment_method)
                            if "error" in order_result:
                                return f"Sorry, I couldn't create your order: {order_result['error']}"
                            elif "success" in order_result:
                                # Get product name from items list
                                product_name = order_result['order']['items'][0]['name'] if order_result['order']['items'] else current_product_name
                                response = (f"Great news! Your order has been confirmed:\n\n"
                                          f"Order ID: {order_result['order']['order_id']}\n"
                                          f"Product: {product_name}\n"
                                          f"Total Price: ${order_result['order']['charges']['total']}\n\n"
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
                        except Exception as e:
                            print(f"Error creating order: {str(e)}")
                            return "I apologize, but there was an error creating your order. Please try again."
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

# --- Billing Agent Class ---
class BillingAgent:
    def __init__(self):
        load_dotenv()
        self.EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
        self.STRIPE_API_KEY = os.getenv('STRIPE_API_KEY')
        stripe.api_key = self.STRIPE_API_KEY
        self.orders_file = ORDERS_FILE  # Use the same orders.json
        self.client = AsyncOpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        set_tracing_disabled(disabled=True)
        self.agent = Agent(
            name="BillingAgent",
            instructions="""
You are a billing agent.
1. First, verify the given order ID via verify_order().
2. If not found, ask the user to provide a valid order ID.
3. Then:
   - For COD payment: call process_cod(order_id).
   - For credit card payment: call process_credit_card(order_id).
4. Return the tool's output directly.
""",
            model=OpenAIChatCompletionsModel(
                model="gemini-2.0-flash",
                openai_client=self.client
            ),
            tools=[self.verify_order, self.generate_invoice, self.send_invoice_email, self.process_cod, self.process_credit_card]
        )

    # Internal logic functions
    def _verify_order(self, order_id: str) -> dict:
        try:
            with open(self.orders_file, 'r', encoding='utf-8') as f:
                orders_data = json.load(f)
            orders = orders_data.get('orders', [])
            order = next((o for o in orders if o['order_id'] == order_id), None)
            if not order:
                return {"error": "Order ID not found."}
            # --- Support new format ---
            if 'customer' in order and 'items' in order:
                return {
                    "order_id": order["order_id"],
                    "date": order.get("order_date", ""),
                    "customer": order["customer"],
                    "items": order["items"],
                    "payment_method": order.get("payment_method", "COD"), # Ensure payment method is passed
                    "subtotal": order.get("subtotal", 0),
                    "tax": order.get("tax", 0),
                    "shipping_fee": order.get("shipping_fee", 0),
                    "handling_fee": order.get("handling_fee", 0),
                    "cod_fee": order.get("cod_fee", 0),
                    "total": order.get("charges", {}).get("total", 0) # Use charges.total
                }
            # --- Fallback to old format ---
            customer = {
                "name": order.get("customer_name", "N/A"),
                "email": order.get("customer_email", "N/A"),
                "address": "N/A"  # Address not available in current data
            }
            items = [{
                "name": order.get("product_name", "N/A"),
                "quantity": 1,
                "unit_price": order.get("price", 0)
            }]
            subtotal = order.get("price", 0)
            tax = round(subtotal * 0.05, 2)  # 5% tax for example
            shipping_fee = 500  # Flat shipping fee (PKR)
            total = subtotal + tax + shipping_fee
            return {
                "order_id": order["order_id"],
                "date": order.get("order_date", ""),
                "customer": customer,
                "items": items,
                "subtotal": subtotal,
                "tax": tax,
                "shipping_fee": shipping_fee,
                "total": total
            }
        except Exception as e:
            return {"error": f"Error reading orders file: {e}"}

    def _generate_invoice(self, order_id: str) -> str:
        order = self._verify_order(order_id)
        if isinstance(order, dict) and order.get('error'):
            return order['error']
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Set global font for consistency
            pdf.set_font("Arial", '', 10)
            
            # Header
            pdf.set_fill_color(230, 230, 250) # Light purple background
            pdf.rect(0, 0, 210, 40, 'F') # Colored band at the top
            pdf.set_text_color(40, 40, 120) # Dark blue text
            pdf.set_font("Arial", 'B', 28)
            pdf.cell(0, 20, "INVOICE", ln=True, align='C')
            pdf.set_font("Arial", 'I', 12)
            pdf.cell(0, 5, "Your Trusted E-commerce Partner", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_text_color(0, 0, 0) # Reset text color to black
            pdf.set_font("Arial", 'B', 12)
            
            # Invoice Details (Top Right)
            pdf.set_x(120)
            pdf.cell(0, 6, f"Invoice #: {order['order_id']}", ln=True, align='L')
            pdf.set_x(120)
            pdf.cell(0, 6, f"Date: {datetime.strptime(order.get('date', ''), '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d')}", ln=True, align='L')
            pdf.ln(5)
            
            # Customer and Order Information Section
            pdf.set_fill_color(240, 240, 240) # Light grey background for sections
            pdf.rect(10, pdf.get_y(), 190, 8, 'F') # Section header background
            pdf.set_font("Arial", 'B', 12)
            pdf.set_xy(10, pdf.get_y())
            pdf.cell(190, 8, "Customer & Order Information", 0, 1, 'L')
            pdf.ln(2)
            
            pdf.set_font("Arial", '', 10)
            # Use multi_cell for address to handle line breaks
            customer_address_str = (
                f"{order['customer']['address']['street']}\n"
                f"{order['customer']['address']['city']}, {order['customer']['address']['state']}\n"
                f"{order['customer']['address']['country']} {order['customer']['address']['postal_code']}"
            )
            
            col_width = 95
            current_y = pdf.get_y()
            
            # Left column for Customer Info
            pdf.set_xy(10, current_y)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 6, "Customer Details:", 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            pdf.set_x(10)
            pdf.cell(col_width, 6, f"Name: {order['customer']['name']}", 0, 1, 'L')
            pdf.set_x(10)
            pdf.cell(col_width, 6, f"Email: {order['customer']['email']}", 0, 1, 'L')
            pdf.set_x(10)
            pdf.cell(col_width, 6, f"Phone: {order['customer']['phone']}", 0, 1, 'L')
            pdf.set_x(10)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 6, "Address:", 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            pdf.set_x(10)
            pdf.multi_cell(col_width, 5, customer_address_str, 0, 'L')
            
            # Right column for Order Info
            pdf.set_xy(105, current_y)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(col_width, 6, "Order Details:", 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            pdf.set_x(105)
            
            # Format payment method for display
            display_payment_method = order.get('payment_method', 'COD').replace('_', ' ').title()
            if display_payment_method.lower() == 'credit card':
                display_payment_method = 'Online (Credit Card)'

            pdf.cell(col_width, 6, f"Payment Method: {display_payment_method}", 0, 1, 'L')
            pdf.set_x(105)
            pdf.cell(col_width, 6, f"Status: {order.get('status', 'pending').title()}", 0, 1, 'L')
            pdf.set_x(105)
            pdf.cell(col_width, 6, f"Payment Status: {order.get('payment_status', 'pending').title()}", 0, 1, 'L')
            pdf.ln(10)
            
            # Product Details Section
            pdf.set_fill_color(240, 240, 240)
            pdf.rect(10, pdf.get_y(), 190, 8, 'F')
            pdf.set_font("Arial", 'B', 12)
            pdf.set_xy(10, pdf.get_y())
            pdf.cell(190, 8, "Product Details", 0, 1, 'L')
            pdf.ln(2)
            
            # Table Header for products
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(200, 220, 255) # Light blue for header
            pdf.cell(80, 7, "Product Name", 1, 0, 'C', 1)
            pdf.cell(30, 7, "Quantity", 1, 0, 'C', 1)
            pdf.cell(40, 7, "Unit Price (PKR)", 1, 0, 'C', 1)
            pdf.cell(40, 7, "Total (PKR)", 1, 1, 'C', 1)
            
            pdf.set_font("Arial", '', 10)
            pdf.set_fill_color(255, 255, 255) # White for rows
            
            for item in order['items']:
                pdf.cell(80, 10, item['name'], 1, 0, 'L')
                pdf.cell(30, 10, str(item['quantity']), 1, 0, 'C')
                pdf.cell(40, 10, f"{item['unit_price']:.2f}", 1, 0, 'R')
                pdf.cell(40, 10, f"{item['quantity'] * item['unit_price']:.2f}", 1, 1, 'R')
                
                # Try to download and add product image below product details
                if 'image_url' in item and item['image_url']:
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Connection': 'keep-alive'
                        }
                        response = requests.get(item['image_url'], headers=headers, stream=True, timeout=10)
                        response.raise_for_status() # Raise an exception for HTTP errors
                        
                        img_data = BytesIO(response.content)
                        img = Image.open(img_data)
                        
                        # Convert image to RGB if needed and save as JPEG
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        temp_img_path = f"temp_{order['order_id']}_{item['name']}.jpeg"
                        img.save(temp_img_path, 'JPEG')
                        
                        # Add image to PDF, resizing if necessary
                        img_width, img_height = img.size
                        max_img_width = 50
                        max_img_height = 50
                        
                        if img_width > max_img_width or img_height > max_img_height:
                            ratio = min(max_img_width / img_width, max_img_height / img_height)
                            img_width = int(img_width * ratio)
                            img_height = int(img_height * ratio)
                        
                        pdf.image(temp_img_path, x=10, y=pdf.get_y() + 2, w=img_width, h=img_height)
                        pdf.ln(img_height + 5) # Add space after image
                        
                        os.remove(temp_img_path) # Clean up temporary image
                        
                    except requests.exceptions.RequestException as req_err:
                        print(f"Warning: Network/HTTP error while adding image for {item['name']}: {str(req_err)}")
                        pdf.set_font("Arial", 'I', 8)
                        pdf.cell(0, 5, "Image not available", 0, 1, 'L')
                        pdf.set_font("Arial", '', 10)
                    except Exception as e:
                        print(f"Warning: Could not add image for {item['name']}: {str(e)}")
                        pdf.set_font("Arial", 'I', 8)
                        pdf.cell(0, 5, "Image processing error", 0, 1, 'L')
                        pdf.set_font("Arial", '', 10)
                else:
                    pdf.ln(5) # Just a little space if no image
            
            pdf.ln(10)
            
            # Charges Breakdown Section
            pdf.set_fill_color(240, 240, 240)
            pdf.rect(10, pdf.get_y(), 190, 8, 'F')
            pdf.set_font("Arial", 'B', 12)
            pdf.set_xy(10, pdf.get_y())
            pdf.cell(190, 8, "Charges Breakdown", 0, 1, 'L')
            pdf.ln(2)
            
            pdf.set_font("Arial", '', 10)
            charges = order.get('charges', {})
            
            # Use a two-column layout for charges
            charge_label_width = 150
            charge_value_width = 40
            
            pdf.cell(charge_label_width, 7, "Subtotal", 0, 0, 'R')
            pdf.cell(charge_value_width, 7, f"{charges.get('subtotal', 0):.2f} PKR", 1, 1, 'R')
            
            pdf.cell(charge_label_width, 7, f"Tax ({charges.get('tax_rate', '1%')})", 0, 0, 'R')
            pdf.cell(charge_value_width, 7, f"{charges.get('tax', 0):.2f} PKR", 1, 1, 'R')
            
            pdf.cell(charge_label_width, 7, "Shipping Fee", 0, 0, 'R')
            pdf.cell(charge_value_width, 7, f"{charges.get('shipping_fee', 0):.2f} PKR", 1, 1, 'R')
            
            pdf.cell(charge_label_width, 7, "Handling Fee", 0, 0, 'R')
            pdf.cell(charge_value_width, 7, f"{charges.get('handling_fee', 0):.2f} PKR", 1, 1, 'R')
            
            if order.get('payment_method', '').lower() == 'cod':
                pdf.cell(charge_label_width, 7, "COD Fee", 0, 0, 'R')
                pdf.cell(charge_value_width, 7, f"{charges.get('cod_fee', 0):.2f} PKR", 1, 1, 'R')
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.set_fill_color(200, 255, 200) # Greenish fill for total
            pdf.cell(charge_label_width, 10, "Total Amount Due", 1, 0, 'R', 1)
            pdf.cell(charge_value_width, 10, f"{charges.get('total', 0):.2f} PKR", 1, 1, 'R', 1)
            
            pdf.ln(15)
            
            # Footer
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, "Thank you for your business!", 0, 1, 'C')
            pdf.cell(0, 5, "For support, please contact us at support@example.com", 0, 1, 'C')
            pdf.set_text_color(0, 0, 0)
            
            invoice_path = f"invoice_{order['order_id']}.pdf"
            pdf.output(invoice_path)
            return invoice_path
        except Exception as e:
            return f"Error generating invoice: {e}"

    def _send_invoice_email(self, order_id: str, invoice_path: str, thank_you: bool = False, credit_card: bool = False) -> str:
        order = self._verify_order(order_id)
        if isinstance(order, dict) and order.get('error'):
            return order['error']
        try:
            to_email = order['customer']['email']
            subject = "Your Invoice"
            content = "Please find your invoice attached."
            if thank_you:
                content = "Thank you for your order! " + content
            msg = MIMEMultipart()
            msg['From'] = self.EMAIL_ADDRESS
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(content, 'plain'))
            with open(invoice_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(invoice_path)}')
                msg.attach(part)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            return f"Invoice sent to {to_email}"
        except Exception as e:
            return f"Failed to send invoice: {e}"

    # Tool wrappers
    @function_tool
    def verify_order(self, order_id: str) -> dict:
        return self._verify_order(order_id)

    @function_tool
    def generate_invoice(self, order_id: str) -> str:
        return self._generate_invoice(order_id)

    @function_tool
    def send_invoice_email(self, order_id: str, invoice_path: str, thank_you: bool = False, credit_card: bool = False) -> str:
        return self._send_invoice_email(order_id, invoice_path, thank_you, credit_card)

    @function_tool
    def process_cod(self, order_id: str) -> str:
        invoice_path = self._generate_invoice(order_id)
        if not os.path.exists(invoice_path):
            return invoice_path  # Already contains error message
        email_result = self._send_invoice_email(order_id, invoice_path, thank_you=True)

        # Update order payment status in the orders file
        try:
            with open(self.orders_file, 'r+') as f:
                orders_data = json.load(f)
                for o in orders_data['orders']:
                    if o['order_id'] == order_id:
                        o['payment_status'] = 'completed'
                        break
                f.seek(0) # Rewind to the beginning of the file
                json.dump(orders_data, f, indent=2)
                f.truncate() # Truncate the file to the new size
        except Exception as e:
            print(f"Error updating order status for COD: {e}")

        return f"Thank you for your order! {email_result}"

    @function_tool
    def process_credit_card(self, order_id: str) -> str:
        order = self._verify_order(order_id)
        if isinstance(order, dict) and order.get('error'):
            return order['error']
        try:
            unit_amount = int(order['total'] * 100)  # Stripe expects cents
            
            # Create line items with product images
            line_items = []
            for item in order['items']:
                line_item = {
                    'price_data': {
                        'currency': 'usd',  # Use 'usd' for Stripe test mode
                        'product_data': {
                            'name': item['name'],
                            'images': [item.get('image_url', '')] if item.get('image_url') else []
                        },
                        'unit_amount': int(item['unit_price'] * 100),  # Convert to cents
                    },
                    'quantity': item['quantity'],
                }
                line_items.append(line_item)
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=line_items,
                mode='payment',
                success_url='https://example.com/success',
                cancel_url='https://example.com/cancel',
                customer_email=order['customer']['email'],
            )
            webbrowser.open(session.url)
            input("Press Enter after completing payment in the browser...")

            # Update order payment status and method in the orders file
            with open(self.orders_file, 'r+') as f:
                orders_data = json.load(f)
                for o in orders_data['orders']:
                    if o['order_id'] == order_id:
                        o['payment_method'] = 'credit_card'
                        o['payment_status'] = 'completed'
                        break
                f.seek(0) # Rewind to the beginning of the file
                json.dump(orders_data, f, indent=2)
                f.truncate() # Truncate the file to the new size

            invoice_path = self._generate_invoice(order_id)
            email_result = self._send_invoice_email(order_id, invoice_path, thank_you=True, credit_card=True)
            return f"Thank you for this order! {email_result}"
        except Exception as e:
            return f"Error in process_credit_card: {e}"

    async def handle_message(self, message: str, sender: str) -> str:
        import re
        # Try to extract order_id and payment method
        order_id_match = re.search(r'(ORD\d+)', message, re.IGNORECASE)
        payment_method = None
        if 'cod' in message.lower() or 'cash on delivery' in message.lower():
            payment_method = 'cod'
        elif 'credit card' in message.lower():
            payment_method = 'credit_card'
        order_id = order_id_match.group(1) if order_id_match else None
        if order_id and payment_method:
            if payment_method == 'cod':
                return self._process_cod(order_id)
            elif payment_method == 'credit_card':
                return self._process_credit_card(order_id)
        # Fallback to LLM agent
        result = await Runner.run(self.agent, message)
        return result.final_output

    # Add these as internal methods for direct calls
    def _process_cod(self, order_id: str) -> str:
        invoice_path = self._generate_invoice(order_id)
        if not os.path.exists(invoice_path):
            return invoice_path  # Already contains error message
        email_result = self._send_invoice_email(order_id, invoice_path, thank_you=True)

        # Update order payment status in the orders file
        try:
            with open(self.orders_file, 'r+') as f:
                orders_data = json.load(f)
                for o in orders_data['orders']:
                    if o['order_id'] == order_id:
                        o['payment_status'] = 'completed'
                        break
                f.seek(0) # Rewind to the beginning of the file
                json.dump(orders_data, f, indent=2)
                f.truncate() # Truncate the file to the new size
        except Exception as e:
            print(f"Error updating order status for COD: {e}")

        return f"Thank you for your order! {email_result}"

    def _process_credit_card(self, order_id: str) -> str:
        order = self._verify_order(order_id)
        if isinstance(order, dict) and order.get('error'):
            return order['error']
        try:
            unit_amount = int(order['total'] * 100)  # Stripe expects cents
            
            # Create line items with product images
            line_items = []
            for item in order['items']:
                line_item = {
                    'price_data': {
                        'currency': 'usd',  # Use 'usd' for Stripe test mode
                        'product_data': {
                            'name': item['name'],
                            'images': [item.get('image_url', '')] if item.get('image_url') else []
                        },
                        'unit_amount': int(item['unit_price'] * 100),  # Convert to cents
                    },
                    'quantity': item['quantity'],
                }
                line_items.append(line_item)
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=line_items,
                mode='payment',
                success_url='https://example.com/success',
                cancel_url='https://example.com/cancel',
                customer_email=order['customer']['email'],
            )
            webbrowser.open(session.url)
            input("Press Enter after completing payment in the browser...")
            invoice_path = self._generate_invoice(order_id)
            email_result = self._send_invoice_email(order_id, invoice_path, thank_you=True, credit_card=True)
            return f"Thank you for this order! {email_result}"
        except Exception as e:
            return f"Error in process_credit_card: {e}"

# --- Orchestrator Agent ---
class OrchestratorAgent:
    def __init__(self, customer_agent, billing_agent):
        self.customer_agent = customer_agent
        self.billing_agent = billing_agent

    def is_billing_query(self, message: str) -> bool:
        billing_keywords = [
            "invoice", "payment", "pay", "credit card", "cod", "cash on delivery", "bill", "stripe", "send invoice", "receipt"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in billing_keywords)

    async def handle_message(self, message: str, sender: str) -> str:
        if self.is_billing_query(message):
            return await self.billing_agent.handle_message(message, sender)
        else:
            return await self.customer_agent.handle_message(message, sender)

# --- Instantiate Agents ---
customer_agent = CustomerSupportAgent()
billing_agent = BillingAgent()
orchestrator_agent = OrchestratorAgent(customer_agent, billing_agent)

# --- Main CLI ---
def main():
    required_vars = [
        'GEMINI_API_KEY'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return

    print("Orchestrator Agent CLI started...")
    print("Type 'exit' to quit.")
    
    # Collect comprehensive customer details
    print("\nPlease provide your complete details:")
    print("-------------------------------------")
    name = input("Full Name: ").strip()
    while not name:
        print("Name is required!")
        name = input("Full Name: ").strip()
        
    email = input("Email Address: ").strip()
    while not email or '@' not in email:
        print("Valid email is required!")
        email = input("Email Address: ").strip()
        
    phone = input("Phone Number: ").strip()
    while not phone:
        print("Phone number is required!")
        phone = input("Phone Number: ").strip()
        
    print("\nAddress Details:")
    print("---------------")
    address = input("Street Address: ").strip()
    while not address:
        print("Street address is required!")
        address = input("Street Address: ").strip()
        
    city = input("City: ").strip()
    while not city:
        print("City is required!")
        city = input("City: ").strip()
        
    state = input("State/Province: ").strip()
    while not state:
        print("State/Province is required!")
        state = input("State/Province: ").strip()
        
    country = input("Country: ").strip()
    while not country:
        print("Country is required!")
        country = input("Country: ").strip()
        
    postal_code = input("Postal/ZIP Code: ").strip()
    while not postal_code:
        print("Postal/ZIP Code is required!")
        postal_code = input("Postal/ZIP Code: ").strip()
    
    # Create complete customer data structure
    customer_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "address": {
            "street": address,
            "city": city,
            "state": state,
            "country": country,
            "postal_code": postal_code
        },
        "created_at": datetime.now().isoformat(),
        "conversation_history": []
    }
    
    # Save customer data
    _save_customer_data(customer_data)
    print("\nThank you! Your details have been saved successfully.")
    print("You can now start shopping!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = asyncio.run(orchestrator_agent.handle_message(user_input, email))
        print(f"Agent: {response}")

if __name__ == "__main__":
    main() 