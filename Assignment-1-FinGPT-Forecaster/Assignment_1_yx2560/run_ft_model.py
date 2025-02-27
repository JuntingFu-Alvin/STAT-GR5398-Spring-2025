import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import finnhub
import json
from typing import List, Dict, Any
import pandas as pd
from huggingface_hub import login
import sys
import time
import traceback

# Constants for different model formats
LLAMA_B_INST, LLAMA_E_INST = "[INST]", "[/INST]"
LLAMA_B_SYS, LLAMA_E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEEPSEEK_FORMAT = "### {role}:"

SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]:
Prediction: ...
Analysis: ..."""

class FinancialDataFetcher:
    def __init__(self):
        if "FINNHUB_KEY" not in os.environ:
            print("Error: FINNHUB_KEY environment variable not set")
            sys.exit(1)
        self.finnhub_client = finnhub.Client(api_key=os.environ["FINNHUB_KEY"])
        print("FinancialDataFetcher initialized successfully")
    
    def get_company_profile(self, symbol: str) -> str:
        """Get company profile and format it as a prompt."""
        print(f"\nFetching company profile for {symbol}...")
        try:
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            if not profile:
                raise ValueError(f"No profile data found for symbol: {symbol}")
            
            company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
                "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
            
            return company_template.format(**profile)
        except finnhub.FinnhubAPIException as e:
            print(f"Finnhub API error for symbol {symbol}: {str(e)}")
            return f"Error: Unable to fetch company profile for {symbol}. Please verify the stock symbol is correct."
        except ValueError as e:
            print(f"Value error for symbol {symbol}: {str(e)}")
            return f"Error: No company profile found for {symbol}. Please verify the stock symbol is correct."
        except Exception as e:
            print(f"Unexpected error while fetching profile for {symbol}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching company profile for {symbol}"

    def get_stock_data(self, symbol: str, start_date: str, end_date: str, max_retries: int = 5) -> pd.DataFrame:
        """Get stock price data using yfinance with retry logic."""
        print(f"\nFetching stock data for {symbol} from {start_date} to {end_date}...")
        for attempt in range(max_retries):
            try:
                stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if stock_data.empty:
                    raise ValueError(f"No stock data found for symbol: {symbol}")
                return stock_data['Close']
            except ValueError as e:
                print(f"Error: {str(e)}")
                return pd.DataFrame()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch stock data after {max_retries} attempts: {str(e)}")
                    return pd.DataFrame()
        return pd.DataFrame()

    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get company news from Finnhub."""
        print(f"\nFetching news for {symbol} from {start_date} to {end_date}...")
        try:
            time.sleep(1)  # Rate limiting for Finnhub
            news = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            if not news:
                print(f"No news found for symbol {symbol}")
                return []

            formatted_news = []
            for n in news:
                news_date = datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S')
                if news_date <= end_date.replace('-', ''):
                    formatted_news.append({
                        "date": news_date,
                        "headline": n['headline'],
                        "summary": n['summary']
                    })
            print(f"Found {len(formatted_news)} news items")
            return formatted_news
        except finnhub.FinnhubAPIException as e:
            print(f"Finnhub API error for symbol {symbol}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error while fetching news for {symbol}: {str(e)}")
            return []

    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financials from Finnhub."""
        print(f"\nFetching basic financials for {symbol}...")
        try:
            time.sleep(1)  # Rate limiting for Finnhub
            financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            
            if not financials or 'series' not in financials or not financials['series']['quarterly']:
                raise ValueError(f"No financial data found for symbol: {symbol}")
                
            latest = {}
            for metric, values in financials['series']['quarterly'].items():
                if values:
                    latest[metric] = values[-1]['v']
            return latest
        except finnhub.FinnhubAPIException as e:
            print(f"Finnhub API error for symbol {symbol}: {str(e)}")
            return {}
        except ValueError as e:
            print(f"Value error for symbol {symbol}: {str(e)}")
            return {}
        except Exception as e:
            print(f"Unexpected error while fetching financials for {symbol}: {str(e)}")
            return {}
        

class ModelInference:
    def __init__(self, base_model_id: str, lora_weights_path: str):
        if "HF_TOKEN" not in os.environ:
            print("Error: HF_TOKEN environment variable not set")
            sys.exit(1)
            
        print(f"\nInitializing model {base_model_id}")
        print(f"LoRA weights path: {lora_weights_path}")
        
        # Login to Hugging Face
        login(os.environ["HF_TOKEN"])
        print("Successfully logged in to Hugging Face")
        
        self.base_model_id = base_model_id.lower()
        self.is_deepseek = "deepseek" in self.base_model_id
        print(f"Using {'DeepSeek' if self.is_deepseek else 'LLaMA'} format")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            padding_side="right"
        )
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded successfully")
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
        self.model.eval()
        print("Model initialization complete")

    def format_prompt(self, prompt: str) -> str:
        """Format prompt according to model type."""
        if self.is_deepseek:
            # DeepSeek format with clearer delineation
            formatted = f"### System:\n{SYSTEM_PROMPT}\n\n### Human:\n{prompt}\n\n### Assistant:\n"
        else:
            # LLaMA format
            formatted = f"{LLAMA_B_INST}{LLAMA_B_SYS}{SYSTEM_PROMPT}{LLAMA_E_SYS}{prompt}{LLAMA_E_INST}"
        
        print(f"\nPrompt format: {'DeepSeek' if self.is_deepseek else 'LLaMA'}")
        print("Formatted prompt length:", len(formatted))
        print("\nPrompt start:", formatted[:200])  # Print start of prompt for debugging
        return formatted

    def extract_response(self, full_response: str) -> str:
        """Extract model response based on format."""
        print("\nExtracting response from:", full_response[:200])  # Print start of response
        try:
            if self.is_deepseek:
                parts = full_response.split("### Assistant:\n")
                if len(parts) > 1:
                    response = parts[-1].split("### Human:")[0].strip()
                else:
                    response = full_response.split("### Assistant:\n")[-1].strip()
            else:
                response = full_response.split(LLAMA_E_INST)[-1].strip()
            
            print("Extracted response length:", len(response))
            print("Response start:", response[:200] if response else "EMPTY")  # Print start of response
            return response
        except Exception as e:
            print(f"Error in response extraction: {str(e)}")
            print("Full response:", full_response)
            raise

    def generate_prediction(self, prompt: str) -> str:
        """Generate prediction using the model."""
        print("\nStarting prediction generation...")
        formatted_prompt = self.format_prompt(prompt)
        
        print("\nTokenizing prompt...")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        print("Input shape:", inputs['input_ids'].shape)
        
        print("\nGenerating response...")
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Increased from 512
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            print("Generation complete. Output shape:", outputs.shape)
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            traceback.print_exc()
            raise
        
        print("\nDecoding response...")
        try:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Full response length:", len(response))
        except Exception as e:
            print(f"Error during decoding: {str(e)}")
            traceback.print_exc()
            raise
        
        final_response = self.extract_response(response)
        if not final_response.strip():
            print("Warning: Empty response generated!")
        return final_response

def create_analysis_prompt(data_fetcher: FinancialDataFetcher, symbol: str, end_date: datetime, weeks_of_history: int = 3) -> str:
    """Create a complete analysis prompt with historical data."""
    try:
        print(f"\nCreating analysis prompt for {symbol} with {weeks_of_history} weeks of history")
        print(f"Analysis end date: {end_date.strftime('%Y-%m-%d')}")
        start_date = end_date - timedelta(weeks=weeks_of_history)
        
        # Get company profile
        company_profile = data_fetcher.get_company_profile(symbol)
        
        # Get historical data week by week
        prompt_parts = [company_profile]
        
        for week in range(weeks_of_history):
            week_end = end_date - timedelta(weeks=week)
            week_start = week_end - timedelta(weeks=1)
            
            # Format dates
            start_str = week_start.strftime('%Y-%m-%d')
            end_str = week_end.strftime('%Y-%m-%d')
            print(f"\nProcessing week {week + 1}: {start_str} to {end_str}")
            
            # Get stock prices with retry logic
            prices = data_fetcher.get_stock_data(symbol, start_str, end_str)
            if len(prices) < 2:
                print(f"Warning: Insufficient price data for period {start_str} to {end_str}")
                continue
                
            start_price = prices.iloc[0].item()
            end_price = prices.iloc[-1].item()
            price_change = "increased" if end_price > start_price else "decreased"
            
            # Get news
            news = data_fetcher.get_company_news(symbol, start_str, end_str)
            
            # Create week section
            week_prompt = f"\nFrom {start_str} to {end_str}, {symbol}'s stock price "
            week_prompt += f"{price_change} from {start_price:.2f} to {end_price:.2f}. "
            week_prompt += "News during this period are listed below:\n\n"
            
            # Add news
            for item in news[:5]:  # Limit to 5 news items per week
                week_prompt += f"[Headline]: {item['headline']}\n"
                week_prompt += f"[Summary]: {item['summary']}\n\n"
                
            prompt_parts.append(week_prompt)
        
        # Get basic financials
        financials = data_fetcher.get_basic_financials(symbol)
        if financials:
            financial_section = "\n[Basic Financials]:\n"
            financial_section += "\n".join(f"{k}: {v}" for k, v in financials.items())
        else:
            financial_section = "\n[Basic Financials]:\nNo basic financial reported."
        
        prompt_parts.append(financial_section)
        
        # Add final analysis request
        next_week_end = (end_date + timedelta(weeks=1)).strftime('%Y-%m-%d')
        prompt_parts.append(f"\nBased on all the information before {end_date.strftime('%Y-%m-%d')}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. Then make your prediction of the {symbol} stock price movement for next week ({end_date.strftime('%Y-%m-%d')} to {next_week_end}). Provide a summary analysis to support your prediction.")
        
        final_prompt = "\n".join(prompt_parts)
        print(f"\nFinal prompt created. Length: {len(final_prompt)}")
        return final_prompt
    except Exception as e:
        print(f"\nError creating analysis prompt: {str(e)}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Run financial analysis with fine-tuned model')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to analyze')
    parser.add_argument('--lora_weights', type=str, required=True, help='Path to LoRA weights')
    parser.add_argument('--base_model', type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                       choices=[
                           "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                           "meta-llama/Meta-Llama-3.1-8B"
                       ],
                       help='Base model ID')
    parser.add_argument('--weeks_history', type=int, default=3, 
                       help='Number of weeks of historical data to analyze')
    parser.add_argument('--end_date', type=str, 
                       help='End date for analysis in YYYY-MM-DD format. Defaults to current date if not specified')
    
    args = parser.parse_args()
    print(f"\nStarting analysis with parameters:")
    print(f"Symbol: {args.symbol}")
    print(f"Base model: {args.base_model}")
    print(f"Weeks of history: {args.weeks_history}")
    
    try:
        # Parse end date if provided, otherwise use current date
        if args.end_date:
            try:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            except ValueError:
                print("Error: End date must be in YYYY-MM-DD format")
                sys.exit(1)
        else:
            end_date = datetime.now()

        print("\nInitializing components...")
        data_fetcher = FinancialDataFetcher()
        model_inference = ModelInference(args.base_model, args.lora_weights)
        
        print("\nCreating analysis prompt...")
        prompt = create_analysis_prompt(data_fetcher, args.symbol, end_date, args.weeks_history)
        print(f"Prompt created successfully. Length: {len(prompt)}")
        
        print("\nGenerating prediction...")
        prediction = model_inference.generate_prediction(prompt)
        
        print(f"\nAnalysis for {args.symbol}:")
        print("=" * 80)
        if prediction.strip():
            print(prediction)
        else:
            print("Warning: Empty prediction generated!")
            print("\nDebug information:")
            print(f"Prompt length: {len(prompt)}")
            print(f"Model type: {'DeepSeek' if 'deepseek' in args.base_model.lower() else 'LLaMA'}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()