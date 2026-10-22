# 导入正则表达式模块，用来做文本匹配、查找、替换等正则相关操作
import re   
# 从llm_client模块(llm_client.py)导入我们封装好的大模型客户端类
from llm_client import HelloAgentsLLM   
# 从tools模块(tools.py)导入两个工具：工具执行器类ToolExecutor、搜索工具函数search
from tools import ToolExecutor, search  

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
# REACT_PROMPT_TEMPLATE 是 ReAct 智能体的提示模板，用来告诉大模型遵循 Thought、Action、Observation 的格式思考与调用工具；
# 模板预留占位符，运行时填充用户问题、工具返回的观察结果，配合正则解析和 ToolExecutor 完成工具调度。
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

# ReAct智能体类，实现 Thought -> Action -> Observation 循环工具调用逻辑
class ReActAgent:
    """
    ReAct智能体：基于ReAct(Reason+Act)思想，让大模型可以自主选择调用工具，迭代思考直到给出最终答案
    """
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        """
        :param llm_client: 封装好的LLM客户端实例，用来和大模型对话
        :param tool_executor: 工具执行器实例，管理所有可用工具、执行工具调用
        :param max_steps: 最大迭代步数，防止无限循环调用工具，默认最多跑5轮
        """
        self.llm_client = llm_client         # 保存LLM客户端实例
        self.tool_executor = tool_executor  # 保存工具执行器实例
        self.max_steps = max_steps           # 保存最大迭代轮次
        self.history = []                    # 保存每一轮 Action、Observation 对话历史，给模型看

    def run(self, question: str):
        """
        智能体入口方法，接收用户问题，启动ReAct循环推理
        :param question: 用户原始问题
        :return: 最终答案字符串；达到最大步数/异常终止返回None
        """
        self.history = []               # 每次跑新问题，清空历史记录
        current_step = 0                # 当前步数计数器
        # 循环：没有超过最大步数就持续思考+调用工具
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 获取所有注册好的工具描述，拼进提示词给模型看
            tools_desc = self.tool_executor.getAvailableTools()
            # 把历史列表拼接成一段文本，喂给prompt模板
            history_str = "\n".join(self.history)
            # 使用ReAct提示模板，填充工具描述、用户问题、历史观测记录 (.format：字符填充)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)
            # 组装OpenAI标准消息格式
            messages = [{"role": "user", 
                         "content": prompt}]
            
            # 调用LLM，流式输出并拿到模型返回完整文本
            response_text = self.llm_client.think(messages=messages)

            # 如果LLM返回None（请求失败），直接退出循环
            if not response_text:
                print("错误：LLM未能返回有效响应。"); break

            # 正则解析模型输出，拆分 Thought 和 Action
            thought, action = self._parse_output(response_text)

            # 打印模型思考内容
            if thought: print(f"🤔 思考: {thought}")
            # 解析不到合法Action，终止流程
            if not action: print("警告：未能解析出有效的Action，流程终止。"); break

            # 判断是不是结束指令Finish，代表模型已经可以给出最终答案，不用再调用工具
            # .startswith() 是字符串内置方法：检查这个字符串开头是不是 Finish，返回 True / False。
            if action.startswith("Finish"):
                # 提取Finish里包裹的最终回答
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 解析Action，拿到工具名、工具入参
            tool_name, tool_input = self._parse_action(action)
            # 解析失败，把错误信息写入history，进入下一轮让模型重新思考
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式，请检查。"); continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            # 从工具执行器取出对应的工具函数
            tool_function = self.tool_executor.getTool(tool_name)
            # 执行工具；找不到工具则返回错误文本作为观测结果
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"

            print(f"👀 观察: {observation}")
            # 把本轮动作和工具返回的观测结果追加到历史，下一轮给模型参考
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # while循环结束：达到最大步数限制，停止推理
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """
        私有方法：正则解析模型输出文本，提取 Thought 和 Action
        :param text: LLM返回的原始文本
        :return: (thought, action) 二元组
        """
        """
            # 正则匹配Thought：
            # Thought:       匹配固定文字
            # \s*            匹配冒号后面0或多个空格
            # (.*?)          非贪婪捕获思考内容（捕获组1）
            # (?=\nAction:|$)正向先行断言：遇到换行Action:或者文本末尾就停止，不把停止标记抓进结果
            # re.DOTALL      让 . 可以匹配换行，支持多行Thought
            # 正则匹配Action：
            # Action:        匹配固定文字
            # \s*            匹配冒号后面0或多个空格
            # (.*?)          非贪婪捕获Action内容（捕获组2）
            # $              匹配文本末尾，确保Action是最后一段     
        """
        # 匹配Thought：从Thought:后面一直到\nAction: 或者文本结尾停止；re.DOTALL让.可以匹配换行
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # 匹配Action：Action:后面一直到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        # 有匹配结果就取出内容并去掉首尾空白，没有就是None
        # .group(1): 取出第 1 个捕获组，也就是正则里 (.*?) 抓出来的思考文本
        # .strip() ：去掉字符串开头、结尾的空白（空格、换行、tab），清理多余空白。
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        私有方法：解析 Action 格式，形如 Search[华为最新手机]
        :param action_text: 如 "Search[华为最新的手机是哪一款]"
        :return: (工具名, 参数)，匹配失败返回 (None, None)
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        """
        私有方法：专门解析Finish[xxx]，提取括号里的最终答案
        :param action_text: Finish[xxx]
        :return: 括号内的答案字符串，匹配失败返回空串
        """
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""


if __name__ == '__main__':
    # 1. 实例化LLM客户端
    llm = HelloAgentsLLM()
    # 2. 实例化工具执行器，用来管理、注册工具
    tool_executor = ToolExecutor()
    # 定义search工具的描述，告诉模型这个工具是干嘛的
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    # 向工具执行器注册搜索工具：工具名字、描述、工具函数
    tool_executor.registerTool("Search", search_desc, search)
    # 实例化ReAct智能体，传入llm和工具管理器
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    # 用户提问
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    # 启动智能体推理循环
    agent.run(question)
