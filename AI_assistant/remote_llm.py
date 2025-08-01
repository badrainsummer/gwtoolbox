import openai
import os

defalt_model = {'deepseek':'deepseek-chat','chatgpt':'gpt-4o'}

class RemoteLLM:
    def __init__(self, model_type='deepseek', model_name='deepseek-chat', system_prompt=None, api_key=None):
        self.model_type = model_type
        if model_name == "0":
            self.model_name = defalt_model[self.model_type]
        else:
            self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_key = api_key

        if self.model_type=='deepseek':
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        elif self.model_type=='chatgpt':
            self.client = openai.OpenAI(api_key=self.api_key)

        self.history = [{"role": "system", "content": self.system_prompt}]

    def __call__(self, dialogue, stream=False):
        self.history.extend(dialogue)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                stream=stream
            )

            if not stream:
                reply = response.choices[0].message.content.strip()
                self.history = [{"role": "system", "content": self.system_prompt}]
                return reply
            else:
                def stream_reply():
                    reply = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            part = chunk.choices[0].delta.content
                            reply += part
                            yield part
                    self.history = [{"role": "system", "content": self.system_prompt}]
                return stream_reply()

        except Exception as e:
            self.history = [{"role": "system", "content": self.system_prompt}]
            if stream:
                def error_gen():
                    yield f"请求失败：{e}"
                return error_gen()
            else:
                return f"请求失败：{e}"
