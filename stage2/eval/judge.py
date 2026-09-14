import json,re
from stage2.llm.client import LLMClient

def llm_correctness(question, gold_answer, pred_answer, client=None):
    client=client or LLMClient()
    prompt=f'''你是严格的科研问答评测器。比较预测答案和参考答案，评估核心事实是否正确且回答是否切题，不因措辞不同扣分。
返回 JSON：{{"score":0到1之间的小数,"reason":"一句话"}}
Question: {question}\nGold: {gold_answer}\nPrediction: {pred_answer}'''
    raw=client.chat([{'role':'user','content':prompt}],temperature=0,max_tokens=300)
    m=re.search(r'\{.*\}',raw,re.S); obj=json.loads(m.group(0) if m else raw)
    return float(obj.get('score',0)),obj.get('reason','')
