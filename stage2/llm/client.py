import os, requests
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def __init__(self, base=None, key=None, model=None):
        self.base=(base or os.getenv('LLM_API_BASE','https://api.deepseek.com')).rstrip('/')
        self.key=key or os.getenv('LLM_API_KEY','')
        self.model=model or os.getenv('LLM_MODEL','deepseek-chat')
        configured=os.getenv('LLM_MAX_TOKENS','').strip()
        self.max_tokens_override=int(configured) if configured else None
        if self.max_tokens_override is not None and self.max_tokens_override <= 0:
            raise ValueError('LLM_MAX_TOKENS must be a positive integer')
    def effective_max_tokens(self, requested):
        return self.max_tokens_override or requested
    def chat(self, messages, temperature=0.0, max_tokens=1200):
        if not self.key:
            raise RuntimeError('LLM_API_KEY is not set')
        url=self.base if self.base.endswith('/chat/completions') else self.base+'/chat/completions'
        r=requests.post(url,headers={'Authorization':f'Bearer {self.key}','Content-Type':'application/json'},json={
            'model':self.model,'messages':messages,'temperature':temperature,
            'max_tokens':self.effective_max_tokens(max_tokens)
        },timeout=120)
        r.raise_for_status(); data=r.json()
        return data['choices'][0]['message'].get('content','').strip()
