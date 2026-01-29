from app.utils.llm import LLMUtils


system_prompt= """不要自称 DeepSeek,
                        不要宣称任何你没实现的能力(文件上传/联网/免费/128K),
                        不确定就说“不确定”,
                        输出只给内容，不要额外免责声明"""   

class LLMClient:
    @staticmethod
    def get_llm():
        return LLMUtils.init_llm()
    
    def ask(self, prompt: str):
        llm = self.get_llm()
        resp=llm.invoke([{"role":"system","content":system_prompt},
                        {"role":"user","content":prompt}])
        return resp.content
    



    def stream_ask(self, prompt: str):
        llm = self.get_llm()
        message=([{"role":"system","content":system_prompt},
                        {"role":"user","content":prompt}])
        for chunk in llm.stream(message):
            text = getattr(chunk, "content", "")
            if text:
                yield text

