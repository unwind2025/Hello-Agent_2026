# 导入操作系统模块，用于读取环境变量
import os
# 从openai包导入OpenAI客户端，兼容OpenAI接口的大模型都可以用这个客户端调用
from openai import OpenAI
# 从python-dotenv包导入load_dotenv，用来加载本地.env配置文件
from dotenv import load_dotenv
# 类型注解：List列表、Dict字典，提升代码可读性，方便IDE提示
from typing import List, Dict

# 加载 .env 文件里面的环境变量（会把.env里的键值对加载到系统环境变量）
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    """
        【函数栈：__init__ 局部空间（函数结束就销毁）】
        ┌─────────────────────────────────────┐
        │ 入参 model = "deepseek-chat"        │  ← 局部变量，只在__init里看得见
        │ 入参 apiKey = None                  │
        │ 入参 baseUrl = None                 │
        └─────────────────────────────────────┘
                ↓ 执行 self.model = model or ...
                ↓ 把值拷贝到实例对象上

        【堆内存：实例对象 llmClient（长期存在，对象不销毁就一直在）】
        ┌─────────────────────────────────────┐
        │ self（就是这个对象本身）              │
        │ ├─ self.model = "deepseek-chat"     │  ← 实例属性
        │ ├─ self.client = OpenAI(...)        │
        │ └─ ...其他属性                       │
        └─────────────────────────────────────┘

        __new__：造对象（分配内存，返回空实例），是生产者
        __init__：给造好的对象赋值、初始化，是装修工人

        def __new__(cls, *args, **kwargs):
            # cls = 当前类本身，不是实例！
            # 必须返回一个实例对象（通常是 cls 的实例）
            ...

        def __init__(self, *args, **kwargs):
            # self = __new__ 造好传过来的实例
            # 不需要return，只负责填充属性
            ...


        只有实例方法才需要第一个 self；静态方法 @staticmethod、类方法 @classmethod 不需要（或者用 cls）
        def think(self, messages):
            pass
        
        @classmethod
        def build_from_env(cls):
            # cls 代表这个类本身 HelloAgentsLLM，不是实例
            return cls(model="xxx", apiKey="yyy")
        
        def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str | None:
        返回值类型注解可以写成 -> str | None ，代表：返回字符串或者 None
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        :param model: 模型名称/模型ID
        :param apiKey: 大模型服务的API密钥
        :param baseUrl: 接口服务地址（DeepSeek这类兼容OpenAI接口的地址）
        :param timeout: 请求超时时间，单位秒
        """
        # 优先用传入的model参数，没有就读取环境变量 MODEL_ID
        self.model = model or os.getenv("MODEL_ID")
        # 优先传入apiKey，没有就读取环境变量 DEEPSEEK_API_KEY
        apiKey = apiKey or os.getenv("DEEPSEEK_API_KEY")
        # 优先传入baseUrl，没有就读取环境变量 DEEPSEEK_BASE_URL
        baseUrl = baseUrl or os.getenv("DEEPSEEK_BASE_URL")
        # 优先传入timeout，没有就读取环境变量 LLM_TIMEOUT，默认值60秒
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        # 校验：模型ID、密钥、接口地址三者缺一不可，缺少直接抛异常
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 实例化OpenAI兼容客户端，填入密钥、接口地址、超时时间
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str | None:
        # List[Dict[str, str]]：泛型类型注解
        #  messages 是一个列表，列表里面的每一个元素都是字典；字典的 key 是字符串，value 也是字符串。
        """
        调用大语言模型进行思考，并返回其响应。
        :param messages: 对话消息列表，遵循OpenAI消息格式 [{"role":"xxx", "content":"xxx"}, ...]
        :param temperature: 温度参数，控制随机性；0代表尽量确定、少随机
        :return: 模型拼接完成后的完整回答字符串，出错返回 None
        """
        print(f" 正在调用 {self.model} 模型...")
        try:
            # 发起chat completions对话请求，开启流式输出 stream=True
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            # 列表缓存每一段流式返回的文本片段，最后拼接成完整答案
            collected_content = []
            # 迭代流式返回的每一个分片chunk
            for chunk in response:
                # 如果当前分片没有choices内容，跳过
                if not chunk.choices:
                    continue
                # 取出本次分片返回的delta增量内容，为空则赋值空字符串
                content = chunk.choices[0].delta.content or ""
                # 实时打印，end=""不换行，flush=True立刻刷新到控制台，实现打字机效果
                print(content, end="", flush=True) 
                # 把分片文本加入缓存列表
                collected_content.append(content)
                """
                append(x)：加一个元素 x ；-> 按分片保存（知道哪一段是模型一次返回的 chunk）
                extend(x)：把 x 里面所有元素逐个加进来；-> 直接拆成单个字，丢失分片信息
                """
            print()  # 流式输出全部结束后，打印换行，恢复正常换行输出
            # 将缓存列表里所有流式文本分片拼接成一个完整字符串，作为函数返回值交给调用者
            # "".join(xxx) 就是：元素和元素之间不加任何分隔符，直接连起来。
            return "".join(collected_content)     
            
        except Exception as e:
            # 捕获调用过程中所有异常，打印错误信息，返回None
            print(f"❌ 调用LLM API时发生错误: {e}")
            # print("❌ 调用LLM API时发生错误:", e)  # 另一种打印方式，逗号会在输出中自动加空格
            return None

# --- 客户端使用示例 ---
# 当这个文件作为主程序直接运行时，执行下面测试代码
if __name__ == '__main__':
    try:
        # 实例化我们封装的LLM客户端，不传参数，自动从.env读取配置
        llmClient = HelloAgentsLLM()
        
        # 构造对话消息：system系统提示 + user用户提问
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        # 调用think方法发起请求，拿到模型返回完整文本
        responseText = llmClient.think(exampleMessages)
        # 如果成功拿到结果，打印完整响应
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        # 捕获初始化时参数缺失抛出的ValueError并打印
        print(e)

