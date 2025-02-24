from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir = "./pretrained-models"

base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    torch_dtype=torch.float16,   # optional if you have enough VRAM
)
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    cache_dir=cache_dir,
)

model = PeftModel.from_pretrained(
    base_model, 
    '/root/FinGPT/finetuned_models/dow30-202305-202405-llama3.1-8B_202502020308', 
    cache_dir=cache_dir, 
    # offload_folder="./offload2/",
    torch_dtype=torch.float16,
    # offload_buffers=True
)
model = model.eval()

# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

# company_intro = """
# [Company Introduction]:

# Apple Inc is a leading entity in the Technology sector. Incorporated and publicly traded since 1980-12-12, the company has established its reputation as one of the key players in the market. As of today, Apple Inc has a market capitalization of 2809837.86 in USD, with 15634.23 shares outstanding.

# Apple Inc operates primarily in the US, trading under the ticker AAPL on the NASDAQ NMS - GLOBAL MARKET. As a dominant force in the Technology space, the company continues to innovate and drive progress within the industry.

# From 2023-10-25 to 2023-11-01, AAPL's stock price increased from 171.10 to 173.97. Company news during this period are listed below:

# [Headline]: 25 Largest Economies in the World by 2075
# [Summary]: In this article, we will be taking a look at the 25 largest economies in the world by 2075. To skip our detailed analysis, you can go directly to see the 5 largest economies in the world by 2075. In both 2022 and 2023, the global economy has struggled significantly after record inflation enveloped most countries across […]

# [Headline]: India opposition accuses govt of trying to hack lawmakers' iPhones
# [Summary]: Indian opposition leader Rahul Gandhi on Tuesday accused Prime Minister Narendra Modi's government of trying to hack into senior opposition politicians' mobile phones, after they reported receiving warning messages from Apple.  Some of the lawmakers shared screenshots on social media of a notification quoting the iPhone manufacturer as saying: "Apple believes you are being targeted by state-sponsored attackers who are trying to remotely compromise the iPhone associated with your Apple ID".  "Hack us all you want," Gandhi told a news conference in New Delhi, in reference to Modi.

# [Headline]: 39% Of This Apple Insider's Holdings Were Sold
# [Summary]: Looking at Apple Inc.'s ( NASDAQ:AAPL ) insider transactions over the last year, we can see that insiders were net...

# [Headline]: Indian opposition MPs accuse government of trying to hack their iPhones
# [Summary]: Ruling BJP rejects claims of involvement following Apple notifications of possible ‘state-sponsored’ attacks

# [Headline]: Should You Buy These 2 ‘Magnificent Seven’ Stocks Ahead of Earnings? Apple and Nvidia in Focus
# [Summary]: What should investors make of this year’s third-quarter earnings? The Q3 results have been pretty good, with 78% of companies reporting so far beating the forecasts, but stocks are still feeling pressure. One obvious sign of that pressure: the S&P 500 this week hit its lowest point since last May, and is just shy of correction territory. The effect is most clearly seen in the ‘Magnificent Seven,’ a group of Big Tech giants whose gains earlier in the year carried the markets generally – but which

# From 2023-11-01 to 2023-11-07, AAPL's stock price increased from 173.97 to 181.25. Company news during this period are listed below:

# [Headline]: Apple Earnings: Why Guidance Will Be Key
# [Summary]: Tech giant Apple (NASDAQ: AAPL) is scheduled to report its fiscal fourth-quarter results on Thursday.  After all, the company's approximately $2.7 trillion market cap is big enough to influence major market indexes like the S&P 500; Apple represents about 7% of the index.  While the company's fiscal fourth-quarter financial performance will definitely be important, investors may pay even closer attention to another metric: management's guidance for its fiscal first-quarter revenue.

# [Headline]: Analysts offer hot takes on Q4 2023 Apple results
# [Summary]: Analysts have weighed in on Apple's Q4 2023 financial results, with most taking the view that the quarter is decent-performing, but with caution about a shorter Q1 2024.

# [Headline]: How to run new macOS versions on older Macs with OpenCore
# [Summary]: Apple removes support for old Mac hardware in new macOS releases. Here's how to run modern macOS on older Macs using OpenCore.

# [Headline]: Apple Watch import ban: what you need to know
# [Summary]: There is a possibility of an import ban in the U.S. on the Apple Watch. Here's what you need to know before it potentially goes into effect on Christmas Day, 2023.

# [Headline]: ChatGPT: Everything you need to know about the AI-powered chatbot
# [Summary]: ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. While there is a more…nefarious side to ChatGPT, it’s clear that AI tools are not going away anytime soon. Since its initial launch nearly a year ago, ChatGPT has hit 100 million weekly active users, and OpenAI i

# [Basic Financials]:

# No basic financial reported.

# Based on all the information before 2023-11-08, let's first analyze the positive developments and potential concerns for AAPL. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. Then make your prediction of the AAPL stock price movement for next week (2023-11-08 to 2023-11-15). Provide a summary analysis to support your prediction.
# """

# prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + company_intro + E_INST

prompt = """
[INST]<<SYS>> 
You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n
<</SYS>> 

[Company Introduction]:

CrowdStrike Holdings Inc is a leading entity in the Technology sector. Incorporated and publicly traded since 2019-06-12, the company has established its reputation as one of the key players in the market. As of today, CrowdStrike Holdings Inc has a market capitalization of 92347.01 in USD, with 246.31 shares outstanding.

CrowdStrike Holdings Inc operates primarily in the US, trading under the ticker CRWD on the NASDAQ NMS - GLOBAL MARKET. As a dominant force in the Technology space, the company continues to innovate and drive progress within the industry.

From 2024-04-21 to 2024-04-28, CRWD's stock price increased from 282.64 to 304.07. News during this period are listed below:

[Headline]: 3 Tech Stocks With Massive Potential That Billionaire Investors Love
[Summary]: These aren't the biggest tech stocks in the market. But they could be someday.

[Headline]: 3 AI Stocks to Buy on the Dip: April 2024
[Summary]: Artificial intelligence (AI) is one of the most exciting markets right now due to the seemingly unlimited potential of its applications in businesses everywhere. Many large companies are making massive investments to integrate AI into their platforms and services to stay ahead of the charge and offer customers the latest and greatest edition tech has to offer. The three stocks we will examine today are currently on a dip and trading at a lower price than their actual value. When stocks dip due t

[Headline]: Ithaka Group Q1 2024 U.S. Growth Commentary
[Summary]: During the first quarter, Ithaka芒聙聶s portfolio outperformed in a strong market, rising 14.9% (gross of fees) vs the R1G rising 11.4%.

[Headline]: 7 A-Rated Tech Stocks For Your Must-Own List
[Summary]: If you want to beat the market these days, you need to be investing in A-rated tech stocks. The technology sector, by far, has been the best stock market sector for the last several quarters, thanks to the advances in artificial intelligence and machine learning. Despite a recent pullback, the tech-heavy Nasdaq composite is up more than 30% in the last 12 months, topping both the S&P 500 (22% gain) and the Dow Jones Industrial Average (13% gain). A-rated tech stocks make ideal picks for growth i

[Headline]: Is CrowdStrike (CRWD) a Buy as Wall Street Analysts Look Optimistic?
[Summary]: According to the average brokerage recommendation (ABR), one should invest in CrowdStrike (CRWD). It is debatable whether this highly sought-after metric is effective because Wall Street analysts' recommendations tend to be overly optimistic. Would it be worth investing in the stock?

From 2024-04-28 to 2024-05-05, CRWD's stock price increased from 304.07 to 310.21. News during this period are listed below:

[Headline]: 3 Growth Stocks That Could Be Multibaggers in the Making: April Edition
[Summary]: Investors spend the extra effort to research growth stocks and construct their portfolios with the hopes of outperforming the stock market. It鈥檚 possible for some growth stocks to become multibaggers for patient investors. However, these same stocks may go through sharp declines on the way to their full potential. Investors should look for corporations that exhibit impressive financial growth, rising profit margins and competitive moats. Stocks with these three factors have a greater chance of o

[Headline]: CrowdStrike Holdings (CRWD) Stock Drops Despite Market Gains: Important Facts to Note
[Summary]: CrowdStrike Holdings (CRWD) closed the most recent trading day at $304.04, moving -0.01% from the previous trading session.

[Headline]: 30 Largest Software Companies in the World by Market Cap
[Summary]: In this article, we will look into the 30 largest software companies in the world by market cap. If you want to skip our detailed analysis, you can go directly to the 5 Largest Software Companies in the World by Market Cap. Software Industry Outlook According to a report by Morningstar, the total revenue of [鈥

[Headline]: CrowdStrike Named Overall Leader in KuppingerCole Identity Threat Detection and Response (ITDR) Leadership Compass
[Summary]: AUSTIN, Texas, April 30, 2024--CrowdStrike (Nasdaq: CRWD) today announced that it has been named an Overall Leader in the KuppingerCole Leadership Compass, Identity Threat Detection and Response (ITDR) 2024: IAM Meets the SOC. CrowdStrike earned a Leader position in every category: Product, Innovation and Market, positioned furthest to the right and highest in Innovation among all vendors evaluated, achieving the overall highest position in the report. This report reinforces CrowdStrike鈥檚 global

[Headline]: Amazon Bets Big With CrowdStrike on Cybersecurity Products
[Summary]: (Bloomberg) -- Amazon.com Inc. is betting big on cybersecurity firm CrowdStrike Holdings Inc., replacing other defensive tools with the company鈥檚 safeguards. Most Read from BloombergSaudi Arabia Steps Up Arrests Of Those Attacking Israel OnlineUS and Saudis Near Defense Pact Meant to Reshape Middle EastBiden Calls Ally Japan 鈥榅enophobic鈥?Along With China, RussiaHuawei Secretly Backs US Research, Awarding Millions in PrizesTurkey Halts All Trade With Israel Over War in GazaAmazon鈥檚 profit-driving

From 2024-05-05 to 2024-05-12, CRWD's stock price increased from 310.21 to 320.76. News during this period are listed below:

[Headline]: Meet the 2 Best Nasdaq-100 Stocks of the Past Year. They Soared 164% and 219%, and Wall Street Says Both Stocks Are Still Buys
[Summary]: Artificial intelligence stocks Nvidia and CrowdStrike led the Nasdaq-100 higher over the past year.

[Headline]: AWS and CrowdStrike expand cybersecurity partnership
[Summary]: CrowdStrike will adopt more AWS software to help develop its AI security tools.

[Headline]: CrowdStrike Announces the Falcon Next-Gen SIEM ISV Ecosystem, Open to Integrating the Most Third-Party Data Sources to Power the AI-Native SOC
[Summary]: AUSTIN, Texas, May 07, 2024--RSA Conference 2024 鈥?May 7, 2024 鈥?CrowdStrike (NASDAQ: CRWD) today announced that CrowdStrike Falcon庐 Next-Gen SIEM now supports the largest ecosystem of ISV data sources of any pure-play cybersecurity vendor. Data from Amazon Web Services (AWS), Cloudflare, Cribl, ExtraHop, Okta, Rubrik, Zscaler and over 500 security and IT leaders can be seamlessly integrated with Falcon platform data, threat intelligence, AI and workflow automation to power the AI-native SOC and

[Headline]: CrowdStrike and Google Cloud Announce Strategic Partnership to Transform AI-Native Cybersecurity
[Summary]: AUSTIN, Texas, May 09, 2024--RSA Conference 2024 鈥?CrowdStrike (Nasdaq: CRWD) today announced an expanded strategic partnership with Google Cloud to power Mandiant鈥檚 Incident Response (IR) and Managed Detection and Response (MDR) services leveraging the CrowdStrike Falcon庐 platform and the Google Cloud Security Operations platform. The partnership focuses on CrowdStrike鈥檚 market-leading Endpoint Detection and Response (EDR), Identity Threat Detection and Response (ITDR) and Exposure Management s

[Headline]: 3 Thrilling Growth Stocks to Grab for the Next Nasdaq Bull Run
[Summary]: Bull runs can lead to outsized gains for patient investors. Some stocks perform better than others, leaving the rest of the market in dust. Growth-oriented investors usually aren鈥檛 shy about taking more risk in exchange for a higher potential upside. However, that doesn鈥檛 mean investors should become reckless. Investors should focus on corporations that offer a good mix of competitive moat, financial growth and rising profit margins. These three growth stocks for the next Nasdaq bull run that ch

Some recent basic financials of CRWD, reported at 2024-04-30, are presented below:

[Basic Financials]:

assetTurnoverTTM: 0.5288
bookValue: 2535.54
cashRatio: 1.379608866312974
currentRatio: 1.8041
ebitPerShare: 0.0277
eps: 0.1712
ev: 68179.729
fcfMargin: 0.3508
fcfPerShareTTM: 4.2127
grossMargin: 0.7557
longtermDebtTotalAsset: 0.1086
longtermDebtTotalCapital: 0.2266
longtermDebtTotalEquity: 0.293
netDebtToTotalCapital: -0.9027
netDebtToTotalEquity: -1.1672
netMargin: 0.0465
operatingMargin: 0.0075
pb: 28.0569
peTTM: 540.0838
pfcfTTM: 69.4407
pretaxMargin: 0.0586
psTTM: 21.6626
ptbv: 29.4754
quickRatio: 1.7337
receivablesTurnoverTTM: 5.6428
roaTTM: 0.0212
roeTTM: 0.0605
roicTTM: 0.0451
rotcTTM: 0.0084
salesPerShare: 3.6817
sgaToSale: 0.2443
tangibleBookValue: 2413.516
totalDebtToEquity: 0.293
totalDebtToTotalAsset: 0.1086
totalDebtToTotalCapital: 0.2266
totalRatio: 1.5888

Based on all the information before 2024-05-12, let's first analyze the positive developments and potential concerns for CRWD. 
Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. 
Then let's assume your prediction for next week (2024-05-12 to 2024-05-19) is up by more than 5%. 
Provide a summary analysis to support your prediction. 
The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.
[/INST]
"""

inputs = tokenizer(
    prompt, 
    return_tensors='pt', 
    max_length=4096, 
    padding=False, 
    truncation=True
)
inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
res = model.generate(
    **inputs, max_length=4096, do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True
)
output = tokenizer.decode(res[0], skip_special_tokens=True)
answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL) # don't forget to import re
print(answer)

# Positive Developments:
# 1. Despite the recent stock market losses, American Express Co. has outperformed its competitors, indicating the company's resilience and strong market position.
# 2. The company is expanding its reach with the opening of a Chase Sapphire Lounge at Boston Airport, which could potentially attract more customers and increase revenue.

# Potential Concerns:
# 1. The profitability of American Express Co. may have peaked, as concerns about the company's future growth have surfaced.
# 2. The company's stock price has been experiencing a downward trend, which could indicate potential investor concerns about the company's future performance.

# Prediction & Analysis:
# Given the current market dynamics, the prediction for the American Express Co. stock price movement for the upcoming week (2023-05-14 to 2023-05-21) is bearish. The company's stock has been experiencing a downward trend, and the recent news about concerns about the company's profitability may further weigh on investor sentiment. However, the company's expansion into new markets and its strong market position could potentially offset these concerns and provide a cushion against the negative sentiment.

# In conclusion, the overall outlook for AXP's stock price movement in the upcoming week is bearish, with a prediction of a slight decrease in the stock price. However, the company's resilience and market position could potentially limit the extent of the decline. It is important to closely monitor the company's performance and financial health in the coming weeks to assess the accuracy of this prediction.

### Standard Answer from Deepseek

# [Positive Developments]:  
# 1. **Strategic Partnerships with Major Cloud Providers**: CrowdStrike expanded collaborations with AWS and Google Cloud, enhancing its AI-native cybersecurity offerings and integrating its Falcon platform into Mandiant鈥檚 services, which could drive adoption and revenue growth.  
# 2. **Industry Recognition**: Named an Overall Leader in KuppingerCole鈥檚 ITDR Leadership Compass, reinforcing its competitive edge in identity threat detection and response.  
# 3. **Amazon鈥檚 Endorsement**: Amazon replaced other cybersecurity tools with CrowdStrike鈥檚 solutions, signaling strong trust and potential enterprise-wide adoption.  

# [Potential Concerns]:  
# 1. **High Valuation Metrics**: Elevated multiples (PE TTM: 540, P/S TTM: 21.66, P/FCF TTM: 69.44) suggest the stock is priced for perfection, leaving little margin for error.  
# 2. **Market Sentiment Risks**: Despite bullish analyst ratings, concerns about over-optimism and a potential pullback after a 13.5% price surge over three weeks could trigger profit-taking.  
# 3. **Margin Pressures**: Low net margin (4.65%) and rising competition in AI-driven cybersecurity may strain profitability despite strong top-line growth.  

# [Prediction & Analysis]:  
# CrowdStrike鈥檚 recent partnerships with AWS and Google Cloud, coupled with Amazon鈥檚 endorsement, highlight its growing dominance in AI-native cybersecurity. These developments are likely to fuel investor optimism, especially as the broader tech sector rallies on AI tailwinds. The company鈥檚 inclusion in multiple 鈥渢op growth stock鈥?lists and its leadership in critical ITDR markets further bolster its narrative as a must-own tech stock. While valuation remains stretched, momentum-driven buying and sector enthusiasm could override short-term concerns. The stock鈥檚 consistent upward trajectory (rising 13.5% in three weeks) suggests bullish sentiment is intact.  

# **Prediction**: CrowdStrike鈥檚 stock price will rise by **more than 5%** in the upcoming week (2024-05-12 to 2024-05-19), driven by continued positive sentiment around its partnerships, AI leadership, and sector tailwinds. However, investors should monitor valuation metrics and broader market conditions for signs of volatility.