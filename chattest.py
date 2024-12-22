from fastapi import APIRouter
from typing import List, Dict, Optional
from schemas import ChatRequest
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import re
import pandas as pd

# First, update the Pydantic models to include analysis
class ComponentAnalysis(BaseModel):
    budget_score: float
    power_score: float
    performance_score: float
    utilization: float

class RecommendedProduct(BaseModel):
    product_id: int
    product_name: str
    category: str
    sales_price: float
    stock_count: Optional[int] = None

class ChatResponse(BaseModel):
    message: str
    recommended_products: List[RecommendedProduct]
    total_price: float
    total_budget: float
    budget_allocation: Dict[str, float]
    build_analysis: Optional[Dict[str, float]] = None  # Add overall analysis

router = APIRouter(prefix="/testchat", tags=["Test Chat"])

load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Load data
SCRIPT_DIR = Path(__file__).parent  # Gets routers directory
df = pd.read_csv(SCRIPT_DIR / "finalbuilds.csv")

# Setting up the chat model
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             google_api_key=GEMINI_API_KEY, 
                             temperature=0.1,
                             streaming=True)

def optimize_budget_distribution(
    total_budget: float,
    categories: List[str] = ['cpu', 'motherboard', 'ram', 'ssd', 'hdd', 'gpu', 'case', 'psu', 'fan', 'cooler']
) -> Dict[str, float]:
    """Distribute budget across components"""
    distributions = {
        'cpu': 0.2,
        'motherboard': 0.1,
        'ram': 0.1,
        'ssd': 0.07,
        'hdd': 0.025,
        'gpu': 0.35,
        'case': 0.05,
        'psu': 0.05,
        'fan': 0.025,
        'cooler': 0.03
    }
    
    return {cat: round(total_budget * distributions[cat], 2) for cat in categories}

def analyze_component_aspects(current_components: List[Dict], candidate: Dict, total_budget: float, current_total: float) -> Dict[str, float]:
    """Analyze different aspects of a component using separate LLM calls"""
    try:
        components_str = "\n".join([f"- {c['category'].upper()}: {c['product_name']} (RM{c['sales_price']})" for c in current_components])
        candidate_price = float(candidate['sales_price'])
        new_total = current_total + candidate_price
        utilization_percentage = (new_total / total_budget) * 100
        
        # Check 1: Budget Utilization
        budget_prompt = f"""Analyze ONLY budget utilization:
Total Budget: RM{total_budget}
Current Spent: RM{current_total}
This Component Cost: RM{candidate_price}
New Total: RM{new_total}
Current Utilization: {utilization_percentage:.1f}%

STRICT REQUIREMENTS:
- Target: 80-99% of total budget
- Must not exceed 99% (RM{total_budget * 0.99})
- Must reach at least 80% (RM{total_budget * 0.80})

Return ONLY a score from 0.0 to 1.0 where:
1.0 = Perfect budget utilization (80-99%)
0.5 = Suboptimal but acceptable
0.0 = Outside target range"""

        # Check 2: Power and Compatibility
        power_prompt = f"""Analyze ONLY power and compatibility:
Current Build:
{components_str}

Candidate {candidate['category'].upper()}:
- {candidate['product_name']} (RM{candidate_price})

Return ONLY a score from 0.0 to 1.0"""

        # Check 3: Performance Balance
        performance_prompt = f"""Analyze ONLY performance balance:
Current Build:
{components_str}

Candidate {candidate['category'].upper()}:
- {candidate['product_name']} (RM{candidate_price})

Return ONLY a score from 0.0 to 1.0"""

        scores = {
            'budget': float(llm.invoke(budget_prompt).content.strip()),
            'power': float(llm.invoke(power_prompt).content.strip()),
            'performance': float(llm.invoke(performance_prompt).content.strip())
        }
        
        # Weight the scores (budget gets higher priority)
        weighted_score = (
            scores['budget'] * 0.4 +      # 40% weight for budget
            scores['power'] * 0.3 +       # 30% weight for power/compatibility
            scores['performance'] * 0.3    # 30% weight for performance
        )
        
        return {
            'scores': scores,
            'weighted_score': weighted_score,
            'utilization': utilization_percentage
        }
        
    except Exception as e:
        print(f"Analysis error for {candidate['product_name']}: {str(e)}")
        return {
            'scores': {'budget': 0.5, 'power': 0.5, 'performance': 0.5},
            'weighted_score': 0.5,
            'utilization': 0
        }

def find_one_component_in_budget(
    category: str, 
    max_price: float,
    current_components: List[Dict] = [],
    total_budget: float = 0,
    current_total: float = 0
) -> Dict:
    """Find best component using multiple LLM checks"""
    filtered_df = df[
        (df['category'].str.lower() == category.lower()) & 
        (df['sales_price'] <= max_price)
    ]
    
    if filtered_df.empty:
        return None
    
    candidates = filtered_df.nlargest(3, 'sales_price')
    best_component = None
    best_analysis = None
    best_weighted_score = -1
    
    for _, candidate in candidates.iterrows():
        component_dict = candidate.to_dict()
        analysis = analyze_component_aspects(
            current_components, 
            component_dict,
            total_budget,
            current_total
        )
        
        if analysis['weighted_score'] > best_weighted_score:
            best_weighted_score = analysis['weighted_score']
            best_component = component_dict
            best_analysis = analysis
    
    if best_component and best_analysis:
        return {
            'product_id': int(best_component['product_id']),
            'product_name': str(best_component['product_name']),
            'category': str(best_component['category']),
            'sales_price': float(best_component['sales_price']),
            'stock_count': int(best_component.get('stock_count', 0)),
            'analysis': {
                'budget_score': best_analysis['scores']['budget'],
                'power_score': best_analysis['scores']['power'],
                'performance_score': best_analysis['scores']['performance'],
                'utilization': best_analysis['utilization']
            }
        }
    return None

def find_optimal_components(budget_allocation: Dict[str, float], total_budget: float) -> List[Dict]:
    """Find optimal components using detailed LLM analysis"""
    components = []
    current_total = 0
    
    # Process categories in priority order
    priority_order = ['cpu', 'gpu', 'motherboard', 'ram', 'psu', 'ssd', 'hdd', 'case', 'cooler', 'fan']
    
    for category in priority_order:
        if category not in budget_allocation:
            continue
            
        max_price = budget_allocation[category]
        
        # Get component with detailed analysis
        component = find_one_component_in_budget(
            category=category,
            max_price=max_price,
            current_components=components,
            total_budget=total_budget,
            current_total=current_total
        )
        
        if component:
            components.append(component)
            current_total += float(component['sales_price'])
    
    return components

def generate_build_summary(components: List[RecommendedProduct], total_budget: float, total_price: float) -> str:
    """Generate a detailed build summary using component analysis"""
    
    # Get key components with their scores
    component_details = []
    for c in components:
        if c.category in ['cpu', 'gpu', 'ram']:
            component_details.append(f"{c.category.upper()}: {c.product_name}")
    
    prompt = f"""As a PC building expert, analyze this build:

Build Overview:
{chr(10).join(component_details)}

Budget Analysis:
- Total Budget: RM{total_budget}
- Actual Cost: RM{round(total_price, 2)}
- Utilization: {round((total_price/total_budget)*100, 1)}%

Create a comprehensive 3-part summary:
1. Budget Efficiency: Comment on the budget utilization
2. Performance Analysis: Evaluate the component synergy
3. Use Case: Specify ideal usage scenarios

Keep it professional but enthusiastic."""
    
    try:
        summary = llm.invoke(prompt).content.strip()
        return summary
    except Exception as e:
        return f"""Build Analysis:
Total: RM{round(total_price, 2)} of RM{total_budget} budget
Key Components: {', '.join(component_details)}
Recommended for high-performance computing tasks."""

@router.post("/", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    try:
        budget_match = re.search(r'RM\s*(\d+)', request.message)

        if not budget_match:
            return ChatResponse(
                message="Please specify a budget (e.g., RM4000)",
                recommended_products=[],
                total_price=0,
                total_budget=0,
                budget_allocation={},
                build_analysis={}
            )
            
        total_budget = float(budget_match.group(1))
        budget_allocation = optimize_budget_distribution(total_budget)
        
        selected_components = find_optimal_components(budget_allocation, total_budget)
        
        recommended_products = []
        total_price = 0
        overall_scores = {
            'budget': 0.0,
            'power': 0.0,
            'performance': 0.0
        }
        
        for component in selected_components:
            # Simplified product creation without analysis
            recommended_products.append(RecommendedProduct(
                product_id=int(component['product_id']),
                product_name=component['product_name'],
                category=component['category'],
                sales_price=float(component['sales_price']),
                stock_count=int(component['stock_count'])
            ))
            total_price += float(component['sales_price'])
            
            # Still accumulate scores for build analysis
            analysis = component.get('analysis', {})
            overall_scores['budget'] += analysis.get('budget_score', 0.5)
            overall_scores['power'] += analysis.get('power_score', 0.5)
            overall_scores['performance'] += analysis.get('performance_score', 0.5)
        
        # Calculate average scores
        num_components = len(recommended_products)
        if num_components > 0:
            overall_scores = {k: v/num_components for k, v in overall_scores.items()}
        
        # Add budget utilization to overall analysis
        overall_scores['budget_utilization'] = (total_price / total_budget) * 100

        message = generate_build_summary(
            components=recommended_products,
            total_budget=total_budget,
            total_price=total_price
        )

        return ChatResponse(
            message=message,
            recommended_products=recommended_products,
            total_price=round(total_price, 2),
            total_budget=total_budget,
            budget_allocation=budget_allocation,
            build_analysis=overall_scores
        )
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        return ChatResponse(
            message=str(e),
            recommended_products=[],
            total_price=0,
            total_budget=0,
            budget_allocation={},
            build_analysis={}
        )