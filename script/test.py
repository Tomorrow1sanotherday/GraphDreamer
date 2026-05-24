from openai import OpenAI

client = OpenAI(
    api_key="MAAS369f45faf38a4db59ae7dc6ed954a399",
    base_url="https://legislation-merely-alt-indicator.trycloudflare.com/v1",
    timeout=60.0,  # 重要：设置足够的超时时间
)

# 测试调用
completion = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100
)
print(completion.choices[0].message.content)